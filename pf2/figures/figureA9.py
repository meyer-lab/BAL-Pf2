"""
Figure A4: XXX
"""
import anndata
from pf2.figures.common import getSetup, subplotLabel
from pf2.tensor import correct_conditions

from pf2.data_import import  add_obs
from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr
from .commonFuncs.plotGeneral import rotate_xaxis
from matplotlib.axes import Axes
import anndata
from scipy.cluster.hierarchy import linkage

import networkx as nx

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((30, 30), (2, 2))

    X = read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    X.uns["Pf2_A"] = correct_conditions(X)

    condition_factors_df = pd.DataFrame(
        data=X.uns["Pf2_A"],
        columns=[f"Cmp. {i}" for i in np.arange(1, X.uns["Pf2_A"].shape[1] + 1)],
    )
    
    partial_correlation_matrix(condition_factors_df, f)

    return f


def partial_correlation_matrix(df: pd.DataFrame, f):
    """Plots partial correlation matrix"""
    cov_DF = df.cov()
    Vi = np.linalg.pinv(cov_DF, hermitian=True) 
    Vi_diag = Vi.diagonal()
    D = np.diag(np.sqrt(1 / Vi_diag))
    pCor = -1 * (D @ Vi @ D) 
    pCor[np.diag_indices_from(pCor)] = 1
    pCorr_DF = pd.DataFrame(pCor, columns=cov_DF.columns, index=cov_DF.columns)
    print(pCorr_DF)

    # cmap = sns.color_palette("vlag", as_cmap=True)
    # f = sns.clustermap(pCorr_DF, robust=True, vmin=-1, vmax=1, 
    #                 #    row_cluster=True, 
    #                 #    col_cluster=True, 
    #                 #    annot=True, 
    #                    cmap=cmap, figsize=(25, 25))
    
    return pCorr_DF
    

