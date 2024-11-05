"""
Figure A11:
"""

import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.axes import Axes
import anndata
from .common import subplotLabel, getSetup
from ..figures.commonFuncs.plotGeneral import bal_combine_bo_covid, rotate_xaxis, add_obs_cmp_both_label, add_obs_label, plot_avegene_cmps
from ..data_import import add_obs, combine_cell_types
from .commonFuncs.plotFactors import bot_top_genes


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

    genes1 = bot_top_genes(X, cmp=cmp1, geneAmount=4)
    genes2 = bot_top_genes(X, cmp=cmp2, geneAmount=4)
    genes = np.concatenate([genes1, genes2])

    for i, gene in enumerate(genes):
        plot_avegene_cmps(X, gene, ax[i])
        rotate_xaxis(ax[i])

    return f

