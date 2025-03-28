"""
Figure A6f_i
"""

import numpy as np
import anndata
from .common import subplotLabel, getSetup
from .commonFuncs.plotGeneral import rotate_xaxis,  plot_avegene_cmps, plot_pair_gene_factors
from ..data_import import add_obs, combine_cell_types
from ..utilities import bot_top_genes, add_obs_cmp_both_label, add_obs_cmp_unique_two
from .commonFuncs.plotPaCMAP import plot_gene_pacmap, plot_labels_pacmap, plot_wp_pacmap
import matplotlib.colors as mcolors


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((14, 14), (4, 4))

    subplotLabel(ax)

    XX = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(XX, "binary_outcome")
    add_obs(XX, "patient_category")
    X = XX[XX.obs["patient_category"] != "Non-Pneumonia Control"] 
    combine_cell_types(X)

    cmp1 = 55; cmp2 = 67
    pos1 = True; pos2 = True
    threshold = 0.1
    X = add_obs_cmp_both_label(X, cmp1, cmp2, pos1, pos2, top_perc=threshold)
    X = add_obs_cmp_unique_two(X, cmp1, cmp2)
      
    colors = ["black",  "turquoise", "fuchsia", "gainsboro"]
    pal = []
    for i in colors:
        pal.append(mcolors.CSS4_COLORS[i])
        
    plot_labels_pacmap(X, "Label", ax[0], color_key=pal)

    genes1 = bot_top_genes(X, cmp=cmp1, geneAmount=1)
    genes2 = bot_top_genes(X, cmp=cmp2, geneAmount=1)
    genes = np.concatenate([genes1, genes2])

    for i, gene in enumerate(genes):
        plot_gene_pacmap(gene, X, ax[i+1])
        
    for i, cmp in enumerate([cmp1, cmp2]):
        plot_wp_pacmap(X, cmp, ax[i+5], cbarMax=0.4)
        
    plot_pair_gene_factors(X, cmp1, cmp2, ax[7])
        
    X = X[X.obs["Label"] != "Both"] 

    for i, gene in enumerate(genes):
        plot_avegene_cmps(X, gene, ax[i+8])
        rotate_xaxis(ax[i])

    return f

