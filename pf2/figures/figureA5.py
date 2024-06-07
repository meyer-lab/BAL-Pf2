"""Figure A5: XX"""

from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
from .commonFuncs.plotFactors import bot_top_genes
from .commonFuncs.plotGeneral import plot_avegene_per_status
from pf2.data_import import add_obs


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 15), (4, 3))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/bal_partial_fitted.h5ad")

    X = add_obs(X, "binary_outcome")

    genes = bot_top_genes(X, cmp=1, geneAmount=6)

    for i, gene in enumerate(np.ravel(genes)):
        plot_avegene_per_status(X, gene, ax[i])
        rotate_xaxis(ax[i])

    return f


def rotate_xaxis(ax, rotation=90):
    """Rotates text by 90 degrees for x-axis"""
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=rotation)
