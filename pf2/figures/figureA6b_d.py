"""
Figure A15:
"""
from .commonFuncs.plotPaCMAP import plot_labels_pacmap
from ..data_import import combine_cell_types, add_obs
import anndata
from .common import subplotLabel, getSetup
import matplotlib.colors as mcolors
import numpy as np
from .commonFuncs.plotFactors import bot_top_genes
from .commonFuncs.plotGeneral import rotate_xaxis, plot_avegene_cmps, add_obs_cmp_both_label_three, add_obs_label_three  
from .commonFuncs.plotPaCMAP import plot_gene_pacmap


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((12, 12), (4, 4))

    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    combine_cell_types(X)

    cmp1 = 20; cmp2 = 27; cmp3 = 35
    pos1 = True; pos2 = True; pos3 = False
    threshold = 0.5
    X = add_obs_cmp_both_label_three(X, cmp1, cmp2, cmp3, pos1, pos2, pos3, top_perc=threshold)
    X = add_obs_label_three(X, cmp1, cmp2, cmp3)
    
    colors = ["black", "fuchsia", "turquoise", "slateblue", "gainsboro"]
    pal = []
    for i in colors:
        pal.append(mcolors.CSS4_COLORS[i])

    plot_labels_pacmap(X, "Label", ax[0], color_key=pal)

    genes1 = bot_top_genes(X, cmp=cmp1, geneAmount=1)
    genes2 = bot_top_genes(X, cmp=cmp2, geneAmount=1)
    genes3 = bot_top_genes(X, cmp=cmp3, geneAmount=1)
    genes = np.concatenate([genes1, genes2, genes3])
    for i, gene in enumerate(genes):
        plot_avegene_cmps(X, gene, ax[i+1])
        rotate_xaxis(ax[i+1])
        
    for i, gene in enumerate(genes):
        plot_gene_pacmap(gene, X, ax[i+7])
    
    return f


