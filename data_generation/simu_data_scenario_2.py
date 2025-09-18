import numpy as np
import pandas as pd
from math import *

def expit(x):
    return 1 / (1 + np.exp(-x))

SEED = 3
# Function to simulate one RCT population j
def simulate(n, mu, s, omega_dict : dict, beta_dict : dict, sigma2_dict : dict , seed=SEED, true_params =None):
    rng = np.random.default_rng(seed)
    # Covariate
    C = rng.normal(loc=mu, scale=s, size=n)
    pop = []
    for c in C: 
        population_model(c)
        z = np.random.choice(["pop0", "pop1", "pop2"], p=population_model(c))
        pop.append(z)

    data_all_df = pd.DataFrame({"C" : C, "POP": pop})


    data_all = []
    for i in range(3):
        data_all.append(data_all_df[data_all_df["POP"] == 'pop'+str(i)])
        df = data_all[i].copy()
        df["A"] = rng.binomial(1, 0.5, size=len(df))
        M=[]
        for j, c in enumerate(data_all[i]['C']):
            p = mediator_model(omega_dict['omega_'+str(i)], c, df['A'].iloc[j])
            M.append(rng.binomial(1,p, size=1)[0])

        df["M"] = pd.Series(M, index=df.index).to_numpy()

        
        eps = rng.normal(loc=0, scale=np.sqrt(sigma2_dict['pop'+str(i)]), size=len(df))
        
        beta = beta_dict['beta_'+str(i)]
        
        Y = beta[0] + beta[1] * df['C'] + beta[2] * df['M'] + beta[3] * df['A'] + eps
        df['Y'] = Y
        data_all[i] = df
    # Outcome


    try :
        a = true_params["a"]
        astar = true_params["astar"]
        j = true_params["j"]
        k = true_params["k"]
        p = true_params["p"]
        res = 0
        pi =len(data_all[k][data_all[k]['A']==a])/len(data_all[k])
        
        for ind, y in enumerate(data_all[k]['Y']):
            c = data_all[k]['C'].iloc[ind]
            if data_all[k]['A'].iloc[ind]==a:
                
                if data_all[k]['M'].iloc[ind] ==1:
                    res = res +((y/pi) *  (mediator_model(omega_dict['omega_'+str(p)],  c, astar)/mediator_model(omega_dict['omega_'+str(k)], c, a))*(population_model( c, s = j)/population_model( c, s = k)))
                else :
                    res = res +(y/pi) *  ((1-mediator_model(omega_dict['omega_'+str(p)], c, astar))/(1-mediator_model(omega_dict['omega_'+str(k)], c, a))*(population_model( c, s = j)/population_model( c, s = k)))
                
        
        res_ipw = res/len(data_all[j])
        
        


        res =0
        for ind, c in enumerate(data_all[j]["C"]):
            beta = beta_dict['beta_'+str(k)]
            y_1 = beta[0] + beta[1] * c+ beta[2] *1+ beta[3] * a  
            y_0 = beta[0] + beta[1] * c+ beta[2] *0+ beta[3] * a
            omega = omega_dict['omega_'+str(p)]
            p_m = mediator_model(omega, c, astar)
            res = res+ y_1*p_m + y_0*(1-p_m)
        res_gformula  = res/len(data_all[j])

        return res_ipw, res_gformula

        
    
    except :
        print("no true value computed")
        return data_all



def population_model( c, s = None):
    denominateur = exp(0.01+0.04*c)+ exp(0.05+0.05*c) + exp(0.1+0.06*c)
    if s == 0:
        return  exp(0.01+0.04*c)/denominateur 
    if s ==1 :
        return  exp(0.05+0.05*c)/denominateur
    if s==2 :
        return exp(0.1+0.06*c)/denominateur
    if s is None:
        return  [exp(0.01+0.04*c)/denominateur ,exp(0.05+0.05*c)/denominateur, exp(0.1+0.06*c)/denominateur]

def mediator_model(omega, c, a):
    return expit(omega[0]+omega[1]*c + omega[2]*a)

