"""Figure A1: Condition, eigen-state, and gene factors, 
along with PaCMAP labeled by cell type"""

import anndata
import pandas as pd
from ..tensor import correct_conditions
from ..data_import import combine_cell_types, add_obs
from ..figures.common import getSetup, subplotLabel
from ..figures.commonFuncs.plotFactors import (
    plot_condition_factors,
    plot_gene_factors,
    plot_eigenstate_factors,
)
from ..figures.commonFuncs.plotPaCMAP import plot_labels_pacmap
from ..figures.commonFuncs.plotGeneral import bal_combine_bo_covid


def makeFigure():
    ax, f = getSetup((20, 20), (3, 3))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    X.uns["Pf2_A"] = correct_conditions(X)
    add_obs(X, "patient_category")
    add_obs(X, "binary_outcome")
    plot_condition_factors(X, ax[0], cond="sample_id", cond_group_labels=pd.Series(label_all_samples(X)))
    ax[0].yaxis.set_ticklabels([])
    plot_eigenstate_factors(X, ax[1])
    plot_gene_factors(X, ax[2])
    ax[2].yaxis.set_ticklabels([])
 
    df = X.obs[["patient_category", "binary_outcome"]].reset_index(drop=True)
    df = bal_combine_bo_covid(df)
    X.obs["Status"] = df["Status"].to_numpy()
    plot_labels_pacmap(X, "Status", ax[3])

    combine_cell_types(X)
    plot_labels_pacmap(X, "cell_type", ax[4])
    plot_labels_pacmap(X, "combined_cell_type", ax[5])

    return f


def label_all_samples(X: anndata.AnnData):
    """Label all patient samples by C19 and lived status"""
    bo_only = ["" for x in range(len(pd.unique(X.obs["sample_id"])))]
    pc_only = ["" for x in range(len(pd.unique(X.obs["sample_id"])))]
    labels_samples = ["" for x in range(len(pd.unique(X.obs["sample_id"])))]

    for i, sample in enumerate(pd.unique(X.obs["sample_id"])):
        bo = pd.unique(X[X.obs.sample_id.isin([sample])].obs["binary_outcome"])
        if bo == 0:
            bo = "L-" 
        else:
            bo = "D-"
        bo_only[i] = bo

        pc = pd.unique(X[X.obs.sample_id.isin([sample])].obs["patient_category"]) 
        if pc == "COVID-19":
            pc = "C19"
        else: 
            pc = "nC19"
        pc_only[i] = pc
        
    for i in range(len(labels_samples)):
        labels_samples[i] = bo_only[i]+pc_only[i]
        
        
    return labels_samples



  