import numpy as np
import pandas as pd
import numpy as np
import xgboost as xgb
import itertools
import matplotlib.pyplot as plt

import pickle
import sys
import os

from sklearn.base import BaseEstimator, RegressorMixin

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_generation.simu_data_scenario_3 import simulate


def population_model(data_all, k ,e, e_2):

    def xgb_obj(y_pred, dtrain):
        C = dtrain.get_data()
        if hasattr(C, "toarray"):  # sparse -> dense
            C = C.toarray()
     
        mean1 = np.mean(C * y_pred)
        mean2 = np.mean(C**2 * y_pred)
        grad = 2 * C * (mean1 - e)/len(C) + 2 * (C**2) * (mean2 - e_2)/len(C)
        hess = 2 * (C**2)/len(C) + 2 * (C**4)/len(C)
        return grad, hess
    
    C = data_all[k]["C"]
    y_fake = np.zeros(len(C))
    dtrain = xgb.DMatrix(C, label=y_fake)
    params = {"max_depth": 3, "eta": 0.1, "verbosity": 1}  # pas d'objective ici
    bst = xgb.train(params, dtrain, num_boost_round=50, obj=xgb_obj)

    
    res = lambda c: bst.predict(xgb.DMatrix(pd.DataFrame({"C": [c]})))[0]

    return res
