"""
Figure J4d: COVID-19 Dysfunctional
"""

import anndata
import matplotlib.colors as mcolors
import numpy as np

from ..data_import import add_obs, combine_cell_types
from ..gene_analysis import gsea_overrep_per_cmp
from ..utilities import (add_obs_cmp_unique_three, add_obs_cmp_unique_two, add_obs_cmp_both_label_three,
                         bot_top_genes)
from .common import getSetup, subplotLabel
from .commonFuncs.plotGeneral import (plot_avegene_cmps,
                                      plot_pair_gene_factors, rotate_xaxis)
from RISE.figures.commonFuncs.plotPaCMAP import (plot_gene_pacmap,
                                                 plot_labels_pacmap,
                                                 plot_wp_pacmap)

COMPONENT_1 = 1
COMPONENT_2 = 4
COMPONENT_3 = 6


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((10, 11), (5, 4))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    X = X[X.obs["patient_category"] != "Non-Pneumonia Control"] 
    combine_cell_types(X)

    # X.obsm["weighted_projections"][:, COMPONENT_1 - 1] *= -1
    # X.varm["Pf2_C"][:, COMPONENT_1 - 1] *= -1

    X = add_obs_cmp_both_label_three(
        X, COMPONENT_1, COMPONENT_2, COMPONENT_3
    )
    X = add_obs_cmp_unique_three(
        X, 
        COMPONENT_1, 
        COMPONENT_2,
        COMPONENT_3
    )

    colors = ["fuchsia", "turquoise", "orangered", "gainsboro", "black"]
    pal = []
    for i in colors:
        pal.append(mcolors.CSS4_COLORS[i])
        
    plot_labels_pacmap(X, "Label", ax[0], color_key=pal)

    genes1 = bot_top_genes(X, cmp=COMPONENT_1, geneAmount=2)
    genes2 = bot_top_genes(X, cmp=COMPONENT_2, geneAmount=2)
    genes = np.concatenate([genes1, genes2])

    for i, gene in enumerate(genes):
        plot_gene_pacmap(gene, X, ax[i+1])
        
    for i, cmp in enumerate([COMPONENT_1, COMPONENT_2]):
        plot_wp_pacmap(X, cmp, ax[i+9], cbarMax=0.4)
        
    plot_pair_gene_factors(X, COMPONENT_1, COMPONENT_2, ax[11])
        
    X = X[X.obs["Label"] != "Both"] 

    for i, gene in enumerate(genes):
        plot_avegene_cmps(X, gene, ax[i+12])
        rotate_xaxis(ax[i+12])

    gsea_overrep_per_cmp(
        X,
        COMPONENT_1,
        pos=True,
        enrichr=False,
        output_file=f"output/figureJ4d_{COMPONENT_1}.svg"
    )
    gsea_overrep_per_cmp(
        X,
        COMPONENT_2,
        pos=True,
        enrichr=False,
        output_file=f"output/figureJ4d_{COMPONENT_2}.svg"
    )

    return f