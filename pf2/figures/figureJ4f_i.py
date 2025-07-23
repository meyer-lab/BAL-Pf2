"""
Figure J4f_i: COVID-19 Protective Responses
"""

import anndata
import matplotlib.colors as mcolors
import numpy as np
from RISE.figures.commonFuncs.plotPaCMAP import (
    plot_gene_pacmap,
    plot_labels_pacmap,
    plot_wp_pacmap,
)

from ..data_import import add_obs, combine_cell_types
from ..gene_analysis import gsea_overrep_per_cmp
from ..utilities import add_obs_cmp_both_label, add_obs_cmp_unique_two, bot_top_genes
from .common import getSetup, subplotLabel
from .commonFuncs.plotGeneral import (
    plot_avegene_cmps,
    plot_pair_gene_factors,
    rotate_xaxis,
)

COMPONENT_1 = 3
COMPONENT_2 = 15


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((11, 10), (4, 4))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    X = X[X.obs["patient_category"] != "Non-Pneumonia Control"] 
    combine_cell_types(X)

    # Invert double negative in component 3
    X.obsm["weighted_projections"][:, COMPONENT_1 - 1] *= -1
    X.varm["Pf2_C"][:, COMPONENT_1 - 1] *= -1
    # X.obsm["weighted_projections"][:, COMPONENT_2 - 1] *= -1
    # X.varm["Pf2_C"][:, COMPONENT_2 - 1] *= -1

    threshold = 0.5
    X = add_obs_cmp_both_label(
        X, 
        COMPONENT_1,
        COMPONENT_2,
        top_perc=threshold
    )
    X = add_obs_cmp_unique_two(
        X, 
        COMPONENT_1, 
        COMPONENT_2
    )
      
    colors = ["black", "fuchsia", "turquoise", "gainsboro"]
    pal = []
    for i in colors:
        pal.append(mcolors.CSS4_COLORS[i])
        
    plot_labels_pacmap(X, "Label", ax[0], color_key=pal)

    genes1 = bot_top_genes(X, cmp=COMPONENT_1, geneAmount=1)
    genes2 = bot_top_genes(X, cmp=COMPONENT_2, geneAmount=1)
    genes = np.concatenate([genes1, genes2])

    for i, gene in enumerate(genes):
        plot_gene_pacmap(gene, X, ax[i+1])
        
    for i, cmp in enumerate([COMPONENT_1, COMPONENT_2]):
        plot_wp_pacmap(X, cmp, ax[i+5], cbarMax=0.4)
        
    plot_pair_gene_factors(X, COMPONENT_1, COMPONENT_2, ax[7])
        
    # X = X[X.obs["Label"] != "Both"]

    for i, gene in enumerate(genes):
        plot_avegene_cmps(X, gene, ax[i+8])
        rotate_xaxis(ax[i+8])

    for cmp in [COMPONENT_1, COMPONENT_2]:
        gsea_overrep_per_cmp(
            X,
            cmp,
            pos=True,
            enrichr=False,
            output_file=f"output/figureJ4c_{cmp}.svg"
        )

    return f