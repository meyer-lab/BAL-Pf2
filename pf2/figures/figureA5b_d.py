"""
Figure A11:
"""

import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.axes import Axes
import anndata
from .common import subplotLabel, getSetup
from .commonFuncs.plotGeneral import rotate_xaxis, add_obs_cmp_both_label, add_obs_label, plot_avegene_cmps
from ..data_import import add_obs, combine_cell_types
from .commonFuncs.plotFactors import bot_top_genes
from .commonFuncs.plotPaCMAP import plot_gene_pacmap, plot_labels_pacmap
import matplotlib.colors as mcolors


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((12, 12), (4, 4))

    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    combine_cell_types(X)

    cmp1 = 2; cmp2 = 25
    pos1 = True; pos2 = False
    threshold = 0.5
    X = add_obs_cmp_both_label(X, cmp1, cmp2, pos1, pos2, top_perc=threshold)
    X = add_obs_label(X, cmp1, cmp2)
    
    # X.loc.obs["Label"]
      
    colors = ["black",   "fuchsia", "turquoise", "gainsboro"]
    pal = []
    for i in colors:
        pal.append(mcolors.CSS4_COLORS[i])
        
    plot_labels_pacmap(X, "Label", ax[0], color_key=pal)

    genes1 = bot_top_genes(X, cmp=cmp1, geneAmount=2)
    genes2 = bot_top_genes(X, cmp=cmp2, geneAmount=2)
    genes = np.concatenate([genes1, genes2])

    for i, gene in enumerate(genes):
        plot_avegene_cmps(X, gene, ax[i+1])
        rotate_xaxis(ax[i+1])
        
    # genes1 = bot_top_genes(X, cmp=cmp1, geneAmount=1)
    # genes2 = bot_top_genes(X, cmp=cmp2, geneAmount=1)
    # genes = np.concatenate([genes1, genes2])

    # for i, gene in enumerate(genes):
    #     plot_gene_pacmap(gene, X, ax[i+5])

    return f

