"""
Figure S1: PLSR rank fitting analysis for mortality prediction
"""
import anndata
import cupy
import numpy as np
import pandas as pd

from .common import getSetup, subplotLabel
from ..data_import import add_obs, combine_cell_types
from ..figures.commonFuncs.plotGeneral import plot_all_bulk_pred
from ..tensor import pf2, correct_conditions
from ..data_import import condition_factors_meta
from ..predict import predict_mortality
from sklearn.metrics import accuracy_score, roc_auc_score



def makeFigure():
    ax, f = getSetup((6, 3), (1, 2))
    subplotLabel(ax)
    
    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    combine_cell_types(X)
    X.obs["condition_unique_idxs"] = pd.Categorical(X.obs["condition_unique_idxs"])


    plot_all_bulk_pred(X, ax[0], ax[1])
    
    df_acc = pd.read_csv("pf2/data/accuracies.csv", index_col=0)
    df_auc = pd.read_csv("pf2/data/auc_rocs.csv", index_col=0)
    ranks = np.arange(5, 185, 5)
    
    ax[0].scatter(ranks, df_auc.values)
    ax[0].set(xticks = ranks, ylabel = "Accuracy", xlabel = "Rank")
    ax[1].scatter(ranks, df_acc.values)
    ax[1].set(xticks = ranks, ylabel = "AUC ROC", xlabel = "Rank")
    ax[0].set(xticks = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180])
    ax[1].set(xticks = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180])
    

    
    return f


