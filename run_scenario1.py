import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

import pickle

from gformula_simu import g_formula_parametric, true_value_parametric,IPW
from data_generation.simu_data_scenario_1 import simulate_rct


from pathlib import Path


BASE_DIR = Path("/home/marie-felicia/postdoc_biblio/simulations")
OUT_PATH = BASE_DIR
OUT_PATH_PICKLE = BASE_DIR / "resultats_scenario_1/"

SEED =3
#PARAMS
## Total sample size
N = 1000   # for example
n1 = N // 4
n2 = N // 2
n3 = N // 4


params = [
    dict(n=n1, mu=0, s=3, lambda0=0, lambda1=0.5, lambda2=0.2, 
       beta0=1, beta1=0.5, beta2=2, beta3=1, sigma2=0.12),
    dict(n=n2, mu=2, s=1, lambda0=0, lambda1=0.5, lambda2=-0.8,
       beta0=1, beta1=0.5, beta2=1, beta3=0.8, sigma2=0.12),
    dict(n=n3, mu=-0.5, s=1.5, lambda0=0, lambda1=0.5, lambda2=0.8,
       beta0=1, beta1=0.5, beta2=0.2, beta3=1.2, sigma2=0.12)
    ]

  
# -------- SimulationS--------
n_sim = 50     # 

print("ok")
true_res_nde_0= dict({})
true_res_nie_1 = dict({})

true_res_10= dict({})
true_res_11 = dict({})
true_res_00= dict({})
true_res_01 = dict({})

true_res_nde_1 =dict({})
true_res_nie_0 =dict({})

for k, p in itertools.product(range(3), repeat=2):
    key = f"({k},{p})"
    true_res_nde_0[key] ={}
    true_res_nie_1[key] ={}
    true_res_nde_1[key] ={}
    true_res_nie_0[key] ={}
    true_res_10[key] ={}
    true_res_11[key] ={}
    true_res_00[key] ={}
    true_res_01[key] ={}
        
        
    for j in range(3):
        s = f"s{j}"
        true_res_nde_0[key][s] = true_value_parametric(j, k, p, 1, 0, params) - true_value_parametric(j, k, p, 0, 0, params)
        true_res_nie_1[key][s] = true_value_parametric(j, k, p, 1, 1, params) - true_value_parametric(j, k, p, 1, 0, params)
     

        true_res_nde_1[key][s] = true_value_parametric(j, k, p, 1, 1, params) - true_value_parametric(j, k, p, 0, 1, params)
        true_res_nie_1[key][s] = true_value_parametric(j, k, p, 0, 1, params) - true_value_parametric(j, k, p, 0, 0, params)
     
        true_res_10[key][s] = true_value_parametric(j, k, p, 1, 0, params) 
        true_res_11[key][s] = true_value_parametric(j, k, p, 1, 1, params)   
        true_res_00[key][s] = true_value_parametric(j, k, p, 0, 0, params) 
        true_res_01[key][s] = true_value_parametric(j, k, p, 0, 1, params)  

results_11 = {key: {s : {"G_formula" : [], "IPW" : []} for s in ["s0", "s1", "s2"]} for key in true_res_nde_0}
results_nie_1 = {key: {s : {"G_formula" : [], "IPW" : []} for s in ["s0", "s1", "s2"]} for key in true_res_nde_0}
results_10 = {key: {s : {"G_formula" : [], "IPW" : []} for s in ["s0", "s1", "s2"]} for key in true_res_nde_0}
results_01 = {key: {s : {"G_formula" : [], "IPW" : []} for s in ["s0", "s1", "s2"]} for key in true_res_nde_0}
results_00 = {key: {s : {"G_formula" : [], "IPW" : []} for s in ["s0", "s1", "s2"]} for key in true_res_nde_0}
results_nde_0 = {key: {s : {"G_formula" : [], "IPW" : []} for s in ["s0", "s1", "s2"]} for key in true_res_nde_0}
results_nie_0 = {key: {s : {"G_formula" : [], "IPW" : []} for s in ["s0", "s1", "s2"]} for key in true_res_nde_0}
results_nde_1 = {key: {s : {"G_formula" : [], "IPW" : []} for s in ["s0", "s1", "s2"]} for key in true_res_nde_0}

for i in range(n_sim):
    pop1 = simulate_rct(**params[0], seed= SEED+i)
    pop2 = simulate_rct(**params[1], seed= SEED+i)
    pop3 = simulate_rct(**params[2], seed= SEED+i)
    data_all = [pop1, pop2, pop3]
    

    for k, p in itertools.product(range(3), repeat=2):
        for j in range(3):
            key = f"({k},{p})"
            s = f"s{j}"
            results_10[key][s]["G_formula"].append(g_formula_parametric(j,k,p, 1, 0, data_all))
            results_11[key][s]["G_formula"].append(g_formula_parametric(j,k,p, 1, 1, data_all))

            results_01[key][s]["G_formula"].append(g_formula_parametric(j,k,p, 0, 1, data_all))
            results_00[key][s]["G_formula"].append(g_formula_parametric(j,k,p, 0, 0, data_all))

            results_nde_0[key][s]["G_formula"].append(results_10[key][s]["G_formula"][i]-results_00[key][s]["G_formula"][i])
            results_nie_1[key][s]["G_formula"].append(results_11[key][s]["G_formula"][i]-results_10[key][s]["G_formula"][i])
            results_nde_1[key][s]["G_formula"].append(results_11[key][s]["G_formula"][i]-results_01[key][s]["G_formula"][i])
            results_nie_0[key][s]["G_formula"].append(results_01[key][s]["G_formula"][i]-results_00[key][s]["G_formula"][i])


            results_10[key][s]["IPW"].append(IPW(j,k,p, 1, 0, data_all))
            results_11[key][s]["IPW"].append(IPW(j,k,p, 1, 1, data_all))

            results_01[key][s]["IPW"].append(IPW(j,k,p, 0, 1, data_all))
            results_00[key][s]["IPW"].append(IPW(j,k,p, 0, 0, data_all))

            results_nde_0[key][s]["IPW"].append(results_10[key][s]["IPW"][i]-results_00[key][s]["IPW"][i])
        
            results_nie_1[key][s]["IPW"].append(results_11[key][s]["IPW"][i]-results_10[key][s]["IPW"][i])
            results_nde_1[key][s]["IPW"].append(results_11[key][s]["IPW"][i]-results_01[key][s]["IPW"][i])
        
            results_nie_0[key][s]["IPW"].append(results_01[key][s]["IPW"][i]-results_00[key][s]["IPW"][i])





res_final = {
    "true_res_nde_0" : true_res_nde_0,
    "true_res_nie_1" : true_res_nie_1,
    "true_res_nde_1" : true_res_nde_1,
    "true_res_nie_0" : true_res_nie_0,
    "true_res_00" : true_res_00,
    "true_res_10" : true_res_10,
    "true_res_01" : true_res_01,
    "true_res_11" : true_res_11,
    "results_11" : results_11,
    "results_01" : results_01,
    "results_00" : results_00,
    "results_10" : results_10,
    "results_nde_0" : results_nde_0,
    "results_nie_1" : results_nie_1,
    "results_nde_1" : results_nde_1,
    "results_nie_0" : results_nie_0
}


file_name = OUT_PATH_PICKLE /"res.pkl"
with open(file_name, "wb") as f:
    pickle.dump(res_final,f)

