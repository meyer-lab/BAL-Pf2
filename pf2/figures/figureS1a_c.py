"""
Figure S1: PLSR rank fitting analysis for mortality prediction
"""
import anndata
import cupy
import numpy as np
import pandas as pd

import scp 

from common import getSetup, subplotLabel
from pf2.data_import import add_obs, combine_cell_types
from pf2.figures.commonFuncs.plotGeneral import plot_all_bulk_pred
from pf2.tensor import pf2, correct_conditions
from pf2.data_import import condition_factors_meta
from pf2.predict import predict_mortality
from sklearn.metrics import accuracy_score, roc_auc_score


MEM_POOL = cupy.get_default_memory_pool()

def makeFigure():
    ax, f = getSetup((6, 3), (1, 2))
    subplotLabel(ax)
    
    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    combine_cell_types(X)
    X.obs["condition_unique_idxs"] = pd.Categorical(X.obs["condition_unique_idxs"])
    pf2_ranks = np.arange(5, 185, 5)
    ranks = [2]
    r2xs = pd.Series(0, dtype=float, index=ranks)
    accuracies = pd.Series(0, dtype=float, index=ranks)
    auc_rocs = accuracies.copy(deep=True)
    # plot_all_bulk_pred(X, ax[1])
    
    for rank in ranks:
        XX, r2x = pf2(X, rank, do_embedding=False)
        XX.uns["Pf2_A"] = correct_conditions(XX)
        cond_fact_meta_df = condition_factors_meta(XX)
        probabilities, labels, _ = predict_mortality(
                XX, cond_fact_meta_df, proba=True
            )
        r2xs.loc[rank] = r2x
        accuracies.loc[rank] = accuracy_score(
                labels,
                probabilities.round().astype(int)
            )
        auc_rocs.loc[rank] = roc_auc_score(
                labels,
                probabilities
            )
        
        del pf2_fac
        MEM_POOL.free_all_blocks()
        print(f"Used bytes: {MEM_POOL.used_bytes()}")
        print(f"Total bytes: {MEM_POOL.total_bytes()}")
    
    ax[0].scatter(ranks, auc_rocs)
    ax[0].set(xticks = ranks, ylabel = "AUC ROC", xlabel = "Rank")
    ax[1].scatter(ranks, accuracies)
    ax[1].set(xticks = ranks, ylabel = "Accuracy", xlabel = "Rank")
    ax[0].set(xticks = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180])
    ax[1].set(xticks = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180])
    
    accuracies.to_csv("/u/scratch/a/aramirez/accuracies.csv")
    auc_rocs.to_csv("/u/scratch/a/aramirez/auc_rocs.csv")
    
    return f


