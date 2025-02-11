"""
Figure S1
"""

import pandas as pd
from .common import getSetup, subplotLabel
from ..tensor import correct_conditions
import numpy as np
from pf2.data_import import import_data, condition_factors_meta
from pf2.figures.common import getSetup
from pf2.predict import predict_mortality_all
from pf2.tensor import pf2


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)
    
    # X = import_data()
    # ranks = np.arange(5, 65, 5)
    # ranks = [2, 5]
    # r2xs = pd.Series(0, dtype=float, index=ranks)
    # accuracies = pd.Series(0, dtype=float, index=ranks)
    # for rank in ranks:
    #     XX, r2x = pf2(X, rank, do_embedding=False)
    #     XX.uns["Pf2_A"] = correct_conditions(XX)
    #     cond_fact_meta_df = condition_factors_meta(XX)
    #     acc, _, _ = predict_mortality_all(XX, cond_fact_meta_df, 
    #                                         n_components=2, proba=True)
    #     r2xs.loc[rank] = r2x
    #     accuracies.loc[rank] = acc
    
    # ax[0].plot(ranks, r2xs)
    # ax[0].set(xticks = ranks, ylabel = "R2X", xlabel = "Rank")
    # ax[1].plot(ranks, accuracies,)
    # ax[1].set(xticks = ranks, ylabel = "Accuracy", xlabel = "Rank")
    
    # ax[0].set(xticks=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    # ax[1].set(xticks=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65])


    return f