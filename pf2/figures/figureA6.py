"""
Figure 6
"""

import numpy as np
import anndata
from .common import subplotLabel, getSetup
from .commonFuncs.plotGeneral import rotate_xaxis, add_obs_cmp_both_label, add_obs_label, plot_avegene_cmps, plot_pair_gene_factors
from ..data_import import add_obs, combine_cell_types
from .commonFuncs.plotFactors import bot_top_genes
from .commonFuncs.plotPaCMAP import plot_gene_pacmap, plot_labels_pacmap, plot_wp_pacmap
import matplotlib.colors as mcolors


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((10, 10), (4, 4))

    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    combine_cell_types(X)

    cmp1 = 9; cmp2 = 32
    pos1 = True; pos2 = True
    threshold = 0.5
    X = add_obs_cmp_both_label(X, cmp1, cmp2, pos1, pos2, top_perc=threshold)
    X = add_obs_label(X, cmp1, cmp2)
      
    colors = ["black", "fuchsia", "turquoise", "gainsboro"]
    pal = []
    for i in colors:
        pal.append(mcolors.CSS4_COLORS[i])
        
    plot_labels_pacmap(X, "Label", ax[0], color_key=pal)

    genes1 = bot_top_genes(X, cmp=cmp1, geneAmount=1)
    genes2 = bot_top_genes(X, cmp=cmp2, geneAmount=1)
    genes = np.concatenate([genes1, genes2])

    for i, gene in enumerate(genes):
        plot_gene_pacmap(gene, X, ax[i+1])
        
    plot_pair_gene_factors(X, cmp1, cmp2, ax[7])
        
    X = X[X.obs["Label"] != "Both"] 

    genes1 = bot_top_genes(X, cmp=cmp1, geneAmount=3)
    genes2 = bot_top_genes(X, cmp=cmp2, geneAmount=3)
    genes = np.concatenate([genes1[-3:], genes2[-3:]])
    
    for i, gene in enumerate(genes):
        plot_avegene_cmps(X, gene, ax[i+8], order=["Cmp9", "Cmp32", "NoLabel"])
        rotate_xaxis(ax[i+8])
 
  

    return f

