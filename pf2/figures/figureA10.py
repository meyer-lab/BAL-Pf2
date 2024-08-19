"""
Figure A4: XXX
"""
import anndata
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from ..tensor import correct_conditions
from .common import getSetup
import seaborn as sns
import scanpy as sc
from pf2.data_import import combine_cell_types, add_obs

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((12, 12), (1, 1))

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
   

    condition_factors_df = pd.DataFrame(
        data=X.uns["Pf2_A"],
        columns=[f"Cmp. {i}" for i in np.arange(1, X.uns["Pf2_A"].shape[1] + 1)],
    )
    combine_cell_types(X)

    cmp = 3
    
    wprojs = X.obsm["weighted_projections"]
    # thres_value = 1
    thres_value= 99
    threshold = np.percentile(wprojs, thres_value, axis=0) 
    # subsetX = X[wprojs[:, cmp-1] < threshold[cmp-1], :]
    
    subsetX = X[wprojs[:, cmp-1] > threshold[cmp-1], :]
    
    sns.violinplot(subsetX.obsm["weighted_projections"][:, cmp-1], ax=ax[0])
    
    # sc.tl.rank_genes_groups(subsetX, "combined_cell_type", method="t-test")
    # sc.pl.rank_genes_groups(subsetX, n_genes=30, save="RGG.png")
        
    
    
    return f

    
