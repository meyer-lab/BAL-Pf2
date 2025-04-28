"""Figure 14"""

import anndata
import scanpy as sc
import pandas as pd
import seaborn as sns
import numpy as np
from .commonFuncs.plotGeneral import rotate_xaxis

import numpy as np
import anndata
from .common import subplotLabel, getSetup
from .commonFuncs.plotGeneral import rotate_xaxis, plot_avegene_cmps, plot_pair_gene_factors, plot_avegene_scatter_cmps
from ..data_import import add_obs, combine_cell_types
from ..utilities import bot_top_genes, add_obs_cmp_both_label, add_obs_cmp_unique_two
from .commonFuncs.plotPaCMAP import plot_gene_pacmap, plot_labels_pacmap, plot_wp_pacmap
import matplotlib.colors as mcolors


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((10, 10), (4, 4))

    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    X = X[X.obs["patient_category"] != "Non-Pneumonia Control"] 
    combine_cell_types(X)
    
    cmp1 = 55; cmp2 = 67
    pos1 = True; pos2 = True
    
    threshold = 0.1
    X = add_obs_cmp_both_label(X, cmp1, cmp2, pos1, pos2, top_perc=threshold)
    X = add_obs_cmp_unique_two(X, cmp1, cmp2)

    # genes1 = ["SFN"]
    genes1 = ["SCGB3A2"]
    
    marker_genes = ["SCGB1A1", "FOXJ1", "SCGB3A1", "DNAH5", "TUBA1A", "MUC5AC", "MUC5B", "AGER", "SFTPC", "HOPX", "ABCA3", "PECAM1", "AGR2"]

    X = X[X.obs["Label"] != "Both"] 
    
    for i, geneA in enumerate(genes1):
        for j, geneB in enumerate(marker_genes):
            genes = np.concatenate([[geneA], [geneB]])
            plot_avegene_scatter_cmps(X, genes, ax[j])
    
        
    return f
