import numpy as np
import pandas as pd
from math import *
import statsmodels.api as sm
from scipy.optimize import minimize

from data_generation.simu_data_scenario_3  import simulate
from src.population_model_xgboost import population_model

from scipy.optimize import root

# Logistic (expit) function
def expit(x):
    return 1 / (1 + np.exp(-x))

# g-formula function
def g_formula_parametric(j, k, p, a, astar, data_all):

    n_j = len(data_all[j])

    ones = np.ones(n_j)                    # if M is just an indicator = 1
    A_j = np.full(n_j, a)
    A_star_j = np.full(n_j, astar)

    # Linear regression Y ~ C + M + A on population k
    X_lin = sm.add_constant(data_all[k][["C", "M", "A"]])
    reg_lin = sm.OLS(data_all[k]["Y"], X_lin).fit()
    beta_hat = reg_lin.params
    
    # Compute g_hat_1 and g_hat_0 using covariates from population j
    C_j = data_all[j]["C"]
    g_hat_1 = beta_hat["const"]*ones + beta_hat["C"]*C_j + beta_hat["M"]*ones + beta_hat["A"]*A_j
    g_hat_0 = beta_hat["const"]*ones + beta_hat["C"]*C_j + beta_hat["A"]*A_j 
    
    # Logistic regression M ~ C + A on population p
    X_log = sm.add_constant(data_all[p][["C", "A"]])
    reg_log = sm.Logit(data_all[p]["M"], X_log).fit(disp=False)
    lambda_hat = reg_log.params
    proba_m = expit((lambda_hat["const"]*ones+lambda_hat["C"]*C_j +lambda_hat["A"]*A_star_j ))
    

    gf = np.mean(g_hat_0*(1-proba_m)+g_hat_1*(proba_m))

    return gf


from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor




def IPW_parametric(j, k, p, a, astar, data_all, e=None, e_2=None, clip_status=False, population_algo="parametric"):
    n_k = len(data_all[k])
    n_j = len(data_all[j])

    n = np.sum([len(d) for d in data_all])
    pj = n_j/n
    
    ### Mediator model population k
    M_log_k = sm.add_constant(data_all[k][["C", "A"]])
    reg_log_k = sm.Logit(data_all[k]["M"], M_log_k).fit(disp=False)
    omega_hat_k = reg_log_k.params 

    def mediator_model_population_k(a,c,m, clip_status):
        if m==1:
            r= expit((omega_hat_k["const"]+omega_hat_k["C"]*c+omega_hat_k["A"]*a))
        if m==0:
            r= 1-expit((omega_hat_k["const"]+omega_hat_k["C"]*c+omega_hat_k["A"]*a))
        if clip_status:
            return np.clip(r, 0.001, 1-0.001)
        else: 
            return r

    
    ### Mediator model population k
    M_log_p = sm.add_constant(data_all[p][["C", "A"]])
    reg_log_p = sm.Logit(data_all[p]["M"], M_log_p).fit(disp=False)
    omega_hat_p = reg_log_p.params


    def mediator_model_population_p(a,c,m, clip_status):
        if m==1:
            r = expit((omega_hat_p["const"]+omega_hat_p["C"]*c+omega_hat_p["A"]*a))
        if m==0:
            r = 1-expit((omega_hat_p["const"]+omega_hat_p["C"]*c+omega_hat_p["A"]*a))
        
        if clip_status:
            return np.clip(r, 0.001, 1-0.001)
        else:
            return r

    ### propensity model population k
    A_log_k = sm.add_constant(data_all[k][["C"]])
    reg_log_pi_k = sm.Logit(data_all[k]["A"], A_log_k).fit(disp=False)
    rho_hat_k = reg_log_pi_k.params 


    def propensity_model_k(a,c, clip_status=False):
        if a==1:
            r= expit((rho_hat_k["const"]+rho_hat_k["C"]*c))
        if a==0:
            r = 1-expit((rho_hat_k["const"]+rho_hat_k["C"]*c))

        if clip_status:
            return np.clip(r, 0.001, 1-0.001)
        else: 
            return r



    

    if e is None:
        e = np.mean(data_all[j]["C"])
    if e_2 is None:
        e_2 = np.mean(data_all[j]['C']**(2))

  
    

    if population_algo == "parametric":
        gamma_optim= gamma_without_IPD(data_all=data_all, j=j,k=k, e=e, e_2=e_2, pj=pj)
        print(gamma_optim.success)
        print(gamma_optim)
        gamma = np.array(gamma_optim['x'])
        print('gamma_'+str(j)+str(k))
        
        def model_pop(c):
            return m_exp_model(gamma, c)

    elif population_algo == "xgboost":
        model_pop = population_model(data_all=data_all, k =k,e=e, e_2=e_2)


    res = 0
    for i in range(n_k):
        if data_all[k]["A"].iloc[i]==a:
            c = data_all[k]["C"].iloc[i]
            m = data_all[k]["M"].iloc[i]
            y_w = data_all[k]["Y"].iloc[i]/propensity_model_k(a,c, clip_status=clip_status )
            if j==k:
                y_w = y_w*(mediator_model_population_p(a=astar, c=c,m=m, clip_status=clip_status )/mediator_model_population_k(a=a, c=c,m=m, clip_status=clip_status ))
            else :
                y_w = y_w*(mediator_model_population_p(a=astar, c=c,m=m, clip_status=clip_status  )/mediator_model_population_k(a=a, c=c,m=m, clip_status=clip_status ))* model_pop(c)

            res = res + y_w
    res = res/n_j
    return res


        
def m_exp_model(gamma, c):
    return exp(gamma[0]+gamma[1]*c)



def gamma_without_IPD(data_all, j,k, e, e_2,pj):
    C = np.asarray(data_all[k]['C'], dtype=float)
    
    def obj(gamma):
        g0, g1 = gamma
        w  = np.exp(g0 + g1*C)
        m1 = np.mean(C*w)
        m2 = np.mean((C**2)*w)
        return (m1 - e)**2+ (m2 - e_2)**2

    def grad(gamma):
        g0, g1 = gamma
        w  = np.exp(g0 + g1*C)
        Mw1 = np.mean(C*w)
        Mw2 = np.mean((C**2)*w)
        Mw3 = np.mean((C**3)*w)
        r1  = np.mean(C*w)      - e
        r2  = np.mean((C**2)*w) - e_2
        g0_grad = 2*( r1*Mw1 + r2*Mw2 )
        g1_grad = 2*( r1*Mw2 + r2*Mw3 )
        return np.array([g0_grad, g1_grad], dtype=float)

    res = minimize(obj, x0=np.array([0,0]), jac=grad, method="L-BFGS-B")
    return res  # gamma*, résultats complets    
  


def true_value_parametric(j, k, p, a, astar,params, seed=3):
    rng = np.random.default_rng(seed)
    v= params[k]["beta0"]+params[k]["beta1"]*params[j]['mu'] + params[k]["beta3"]*a

    def h(c):
        return expit(params[p]["lambda0"]+ params[p]["lambda1"]*c+params[p]["lambda2"]*astar )
    
    C_mc = rng.normal(loc=params[j]['mu'], scale=params[j]['s'], size=10000)
    e=np.mean([h(c) for c in C_mc])
    v=v+ params[k]["beta2"]*e
    return v


def true_value_scenario_2(j,k,p,a,astar, params, seed=3):
    rng = np.random.default_rng(seed)

    true_params = dict({'j':j, 'k':k,"p":p, "a":a, "astar":astar})
    res_ipw, res_gformula = simulate(100000,**params,  seed=3, true_params =true_params)

    
    #if abs(res_ipw - res_gformula) >0.05 :
        #print("params to check")
        #print(true_params)
        #print(abs(res_ipw - res_gformula))
        #print("***************")
    return (res_ipw + res_gformula)/2
