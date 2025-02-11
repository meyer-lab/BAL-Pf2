"""
Figure S1
"""

import pandas as pd
import numpy as np
import anndata
from .common import getSetup, subplotLabel
from ..tensor import correct_conditions
from ..data_import import condition_factors_meta
from ..figures.common import getSetup
from ..predict import predict_mortality_all
from ..tensor import pf2


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)
    
    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    X.obs["condition_unique_idxs"] = pd.Categorical(X.obs["condition_unique_idxs"])
    ranks = np.arange(5, 70, 5)
    ranks = [2]
    r2xs = pd.Series(0, dtype=float, index=ranks)
    accuracies = pd.Series(0, dtype=float, index=ranks)
    
    for rank in ranks:
        XX, r2x = pf2(X, rank, do_embedding=False)
        XX.uns["Pf2_A"] = correct_conditions(XX)
        cond_fact_meta_df = condition_factors_meta(XX)
        acc, _, _ = predict_mortality_all(XX, cond_fact_meta_df, 
                                            n_components=2, proba=False)
        r2xs.loc[rank] = r2x
        accuracies.loc[rank] = acc
    
    ax[0].scatter(ranks, r2xs)
    ax[0].set(xticks = ranks, ylabel = "R2X", xlabel = "Rank")
    ax[1].scatter(ranks, accuracies,)
    ax[1].set(xticks = ranks, ylabel = "Accuracy", xlabel = "Rank")
    
    ax[0].set(xticks=[0, 10, 20, 30, 40, 50, 60, 70])
    ax[1].set(xticks=[0, 10, 20, 30, 40, 50, 60, 70])


    return f