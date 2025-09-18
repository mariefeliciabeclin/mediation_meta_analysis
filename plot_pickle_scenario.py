import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from pathlib import Path

import pickle

BASE_DIR = Path("/home/marie-felicia/postdoc_biblio/simulations")
OUT_PATH = BASE_DIR
OUT_PATH_PICKLE = BASE_DIR / "resultats_scenario_2/"

pkl_path = OUT_PATH_PICKLE /"res.pkl"


with open(pkl_path, "rb") as f:
    data = pickle.load(f)

globals().update(data)


# -------- Boxplot --------

col_s = ['red', 'blue', 'green']
methods = ["G_formula", "IPW", "IPW_clip"]
population = ["s0", "s1", "s2"]

colors_map = {
    "G_formula": "#1f77b4",  # bleu
    "IPW": "#ff7f0e",        # orange
}

fig, ax = plt.subplots(3,3, figsize=(15, 25), sharey=True)

for k, p in itertools.product(range(3), repeat=2):
    key = f"({k},{p})"
    d   = results_nde_0[key]
    data   = []
    labels = []
    for s in population: 
        for m in methods: 
            data.append(d[s][m])           # liste de valeurs
            labels.append(f"S={s}-{m}")   

    ax[k, p].boxplot(data, labels=labels)

    # ligne de vérité par s, étendue sur les 2 boîtes s-G et s-IPW
    for i, s in enumerate(population):
        y     = true_res_nde_0[key][s]       # valeur vraie pour s
        pos1  = 2*i + 1                       # s-G
        pos2  = 2*i + 2                       # s-IPW
        ax[k, p].hlines(y=y, xmin=pos1-0.3, xmax=pos2+0.3,
                        colors=col_s[i], linestyles="--")

    ax[k, p].set_title(rf"$\hat{{NDE}}(j, k={k}, p={p}; a^* = 0)$")
    ax[k, p].tick_params(axis="x", rotation=45)



#plt.show()
plt.savefig(OUT_PATH_PICKLE/"nde_0.pdf")


fig, ax = plt.subplots(3,3, figsize=(15, 25), sharey=True)

for k, p in itertools.product(range(3), repeat=2):
    key = f"({k},{p})"
    d   = results_nde_1[key]
    data   = []
    labels = []
    for s in population: 
        for m in methods: 
            data.append(d[s][m])           # liste de valeurs
            labels.append(f"S={s}-{m}")   

    ax[k, p].boxplot(data, labels=labels)

    # ligne de vérité par s, étendue sur les 2 boîtes s-G et s-IPW
    for i, s in enumerate(population):
        y     = true_res_nde_1[key][s]       # valeur vraie pour s
        pos1  = 2*i + 1                       # s-G
        pos2  = 2*i + 2                       # s-IPW
        ax[k, p].hlines(y=y, xmin=pos1-0.3, xmax=pos2+0.3,
                        colors=col_s[i], linestyles="--")

    ax[k, p].set_title(rf"$\hat{{NDE}}(j, k={k}, p={p}; a^* = 1)$")
    ax[k, p].tick_params(axis="x", rotation=45)



#plt.show()
plt.savefig(OUT_PATH_PICKLE/"nde_1.pdf")

fig, ax = plt.subplots(3,3, figsize=(15, 25), sharey=True)

for k, p in itertools.product(range(3), repeat=2):
    key = f"({k},{p})"
    d   = results_nie_1[key]
    data   = []
    labels = []
    for s in population: 
        for m in methods:
            data.append(d[s][m])           # liste de valeurs
            labels.append(f"S={s}-{m}")   

    ax[k, p].boxplot(data, labels=labels)

    # ligne de vérité par s, étendue sur les 2 boîtes s-G et s-IPW
    for i, s in enumerate(population):
        y     = true_res_nie_1[key][s]       # valeur vraie pour s
        pos1  = 2*i + 1                       # s-G
        pos2  = 2*i + 2                       # s-IPW
        ax[k, p].hlines(y=y, xmin=pos1-0.3, xmax=pos2+0.3,
                        colors=col_s[i], linestyles="--")

    ax[k, p].set_title(rf"$\hat{{NIE}}(j, k={k}, p={p}; a = 1)$")
    ax[k, p].tick_params(axis="x", rotation=45)
    
#plt.show()
plt.savefig(OUT_PATH_PICKLE/"nie_1.pdf")


ig, ax = plt.subplots(3,3, figsize=(15, 25), sharey=True)

for k, p in itertools.product(range(3), repeat=2):
    key = f"({k},{p})"
    d   = results_nie_0[key]
    data   = []
    labels = []
    for s in population: 
        for m in methods:
            data.append(d[s][m])           # liste de valeurs
            labels.append(f"S={s}-{m}")   

    ax[k, p].boxplot(data, labels=labels)

    # ligne de vérité par s, étendue sur les 2 boîtes s-G et s-IPW
    for i, s in enumerate(population):
        y     = true_res_nie_0[key][s]       # valeur vraie pour s
        pos1  = 2*i + 1                       # s-G
        pos2  = 2*i + 2                       # s-IPW
        ax[k, p].hlines(y=y, xmin=pos1-0.3, xmax=pos2+0.3,
                        colors=col_s[i], linestyles="--")

    ax[k, p].set_title(rf"$\hat{{NIE}}(j, k={k}, p={p}; a = 0)$")
    ax[k, p].tick_params(axis="x", rotation=45)
    
#plt.show()
plt.savefig(OUT_PATH_PICKLE/"nie_0.pdf")

fig, ax = plt.subplots(3,3, figsize=(15, 25), sharey=True)

for k, p in itertools.product(range(3), repeat=2):
    key = f"({k},{p})"
    d   = results_00[key]
    data   = []
    labels = []
    for s in population: 
        for m in methods: 
            data.append(d[s][m])           # liste de valeurs
            labels.append(f"S={s}-{m}")   

    ax[k, p].boxplot(data, labels=labels)

    # ligne de vérité par s, étendue sur les 2 boîtes s-G et s-IPW
    for i, s in enumerate(population):
        y     = true_res_00[key][s]       # valeur vraie pour s
        pos1  = 2*i + 1                       # s-G
        pos2  = 2*i + 2                       # s-IPW
        ax[k, p].hlines(y=y, xmin=pos1-0.3, xmax=pos2+0.3,
                        colors=col_s[i], linestyles="--")

    latex_text = r"$\hat{{\mathbb{{E}}}} \left[Y(a=0, k={}, M(a^*=0, p={})) \mid S=j\right]$".format(k, p)
    ax[k, p].set_title(latex_text)
    ax[k, p].tick_params(axis="x", rotation=45)
    
#plt.show()
    
 
    

#plt.show()
plt.savefig(OUT_PATH_PICKLE/"00.pdf")






fig, ax = plt.subplots(3,3, figsize=(15, 25), sharey=True)

for k, p in itertools.product(range(3), repeat=2):
    key = f"({k},{p})"
    d   = results_10[key]
    data   = []
    labels = []
    for s in population: 
        for m in methods: 
            data.append(d[s][m])           # liste de valeurs
            labels.append(f"S={s}-{m}")   

    ax[k, p].boxplot(data, labels=labels)

    # ligne de vérité par s, étendue sur les 2 boîtes s-G et s-IPW
    for i, s in enumerate(population):
        y     = true_res_10[key][s]       # valeur vraie pour s
        pos1  = 2*i + 1                       # s-G
        pos2  = 2*i + 2                       # s-IPW
        ax[k, p].hlines(y=y, xmin=pos1-0.3, xmax=pos2+0.3,
                        colors=col_s[i], linestyles="--")

    latex_text = r"$\hat{{\mathbb{{E}}}} \left[Y(a=1, k={}, M(a^*=0, p={})) \mid S=j\right]$".format(k, p)
    ax[k, p].set_title(latex_text)
    ax[k, p].tick_params(axis="x", rotation=45)
    
#plt.show()
    
 
    

#plt.show()
plt.savefig(OUT_PATH_PICKLE/"10.pdf")






fig, ax = plt.subplots(3,3, figsize=(15, 25), sharey=True)

for k, p in itertools.product(range(3), repeat=2):
    key = f"({k},{p})"
    d   = results_11[key]
    data   = []
    labels = []
    for s in population: 
        for m in methods: 
            data.append(d[s][m])           # liste de valeurs
            labels.append(f"S={s}-{m}")   

    ax[k, p].boxplot(data, labels=labels)

    # ligne de vérité par s, étendue sur les 2 boîtes s-G et s-IPW
    for i, s in enumerate(population):
        y     = true_res_11[key][s]       # valeur vraie pour s
        pos1  = 2*i + 1                       # s-G
        pos2  = 2*i + 2                       # s-IPW
        ax[k, p].hlines(y=y, xmin=pos1-0.3, xmax=pos2+0.3,
                        colors=col_s[i], linestyles="--")

    latex_text = r"$\hat{{\mathbb{{E}}}} \left[Y(a=1, k={}, M(a^*=1, p={})) \mid S=j\right]$".format(k, p)
    ax[k, p].set_title(latex_text)
    ax[k, p].tick_params(axis="x", rotation=45)
    
#plt.show()
    
 
    

#plt.show()
plt.savefig(OUT_PATH_PICKLE/"11.pdf")




fig, ax = plt.subplots(3,3, figsize=(15, 25), sharey=True)

for k, p in itertools.product(range(3), repeat=2):
    key = f"({k},{p})"
    d   = results_01[key]
    data   = []
    labels = []
    for s in population: 
        for m in methods: 
            data.append(d[s][m])           # liste de valeurs
            labels.append(f"S={s}-{m}")   

    

    ax[k, p].boxplot(data, labels=labels)

    # ligne de vérité par s, étendue sur les 2 boîtes s-G et s-IPW
    for i, s in enumerate(population):
        y     = true_res_01[key][s]       # valeur vraie pour s
        pos1  = 2*i + 1                       # s-G
        pos2  = 2*i + 2                       # s-IPW
        ax[k, p].hlines(y=y, xmin=pos1-0.3, xmax=pos2+0.3,
                        colors=col_s[i], linestyles="--")

    latex_text = r"$\hat{{\mathbb{{E}}}} \left[Y(a=0, k={}, M(a^*=1, p={})) \mid S=j\right]$".format(k, p)
    ax[k, p].set_title(latex_text)
    ax[k, p].tick_params(axis="x", rotation=45)
    

#plt.show()
plt.savefig(OUT_PATH_PICKLE/"01.pdf")





