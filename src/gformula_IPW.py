import numpy as np
import pandas as pd
from math import *
import statsmodels.api as sm
from scipy.optimize import minimize

from data_generation.simu_data_scenario_2  import simulate

from scipy.optimize import root

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import numpy as np
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score




MODEL_SCIKIT = dict({"logistic_reg" : LogisticRegression, 
                    "linear_reg" :  LinearRegression, 
                    "rf_classifier" :  RandomForestClassifier , 
                    "rf_regressor" :RandomForestRegressor })



def g_formula(j, k, p, a, astar, data_all, outcome_algo, mediator_algo, population_algo):

    def outcome_model(c,m,a):
        model = MODEL_SCIKIT[outcome_algo]()
        model.fit(X=data_all[k][["C", "M", "A"]], y=data_all[k]["Y"])
        return model.predict(np.array([[c, m, a]]))[0]

    def mediator_model(c,a):
        model = MODEL_SCIKIT[mediator_algo]()
        model.fit(X=data_all[p][["C", "A"]], y=data_all[p]["M"])
        return model.predict_proba(np.array([[c, a]]))[0][1] 

    C = data_all[j]["C"]

    g_hat_1 = np.mean([outcome_model(c=c,m=1, a=a)  for c in C])
    g_hat_0 = np.mean([outcome_model(c=c,m=0, a=a)  for c in C])
    
    proba_m = [mediator_model(c=c,a=astar)  for c in C]
    gf = np.mean(g_hat_0*(1-proba_m)+g_hat_1*(proba_m))
    return gf


    def population_model(k)
        def custom_cost(e, e_2, y_pred):
            C= data_all[k]["C"]
            return (np.mean([C[i]*y_pred[i] for i in range(len(C))]) -e)**2 +(np.mean([C[i]**2*(y_pred[i])for i in range(len(C))]) -e_2)**2 


        custom_scorer = make_scorer(custom_cost, greater_is_better=False)


# Modèle
        model = MODEL_SCIKIT[population_algo]()

        def evaluate_model(model, X, y):
            model.fit(X, y)        # entraînement classique
            y_pred = model.predict(X)  
            return custom_objective(y_pred)


# Évaluation avec la fonction de coût custom
        scores = cross_val_score(model, X=C, y, scoring=custom_scorer, cv=5)
        print("Scores avec fonction de coût custom:", scores)




def IPW(j, k, p, a, astar, data_all, e=None, e_2=None, clip_status=False, outcome_algo, mediator_algo, population_algo):
    n_k = len(data_all[k])
    n_j = len(data_all[j])

    n = np.sum([len(d) for d in data_all])
    pj = n_j/n
    
    def mediator_model_population(k,a,c,m, clip_status):
        model = MODEL_SCIKIT[mediator_algo]()
        model.fit(X=data_all[k][["C", "A"]], y=data_all[k]["M"])
        if m==1:
            r= model.predict_proba(np.array([[c, a]]))[0][1]
        if m==0:
            r= 1-model.predict_proba(np.array([[c, a]]))[0][1]
        if clip_status:
            return np.clip(r, 0.001, 1-0.001)
        else: 
            return r

    def propensity_model(k, a,c, clip_status=False):
        model = MODEL_SCIKIT[mediator_algo]()
        model.fit(X=data_all[k][["C", "A"]], y=data_all[k]["M"])
        if a==1:
            r= model.predict_proba(np.array([[c, a]]))[0][1]
        if a==0:
            r = 1-model.predict_proba(np.array([[c, a]]))[0][1]

        if clip_status:
            return np.clip(r, 0.001, 1-0.001)
        else: 
            return r

    if e is None:
        e = np.mean(data_all[j]["C"])
    if e_2 is None:
        e_2 = np.mean(data_all[j]['C']**(2))




    for i in range(n_k):
        if data_all[k]["A"].iloc[i]==a:
            c = data_all[k]["C"].iloc[i]
            m = data_all[k]["M"].iloc[i]
            y_w = data_all[k]["Y"].iloc[i]/propensity_model_k(a,c, clip_status=clip_status )
            if j==k:
                y_w = y_w*(mediator_model_population_p(a=astar, c=c,m=m, clip_status=clip_status )/mediator_model_population_k(a=a, c=c,m=m, clip_status=clip_status ))
            else :
                y_w = y_w*(mediator_model_population_p(a=astar, c=c,m=m, clip_status=clip_status  )/mediator_model_population_k(a=a, c=c,m=m, clip_status=clip_status ))*  m_exp_model(gamma, c)

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



    res = minimize(obj, x0=np.array([0,0]), jac=grad, method="L-BFGS-B")
    return res  # gamma*, résultats complets    
  
