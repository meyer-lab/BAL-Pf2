"""Figure S4"""

from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
from .commonFuncs.plotFactors import bot_top_genes
from .commonFuncs.plotGeneral import plot_avegene_per_status
from ..data_import import add_obs
from .commonFuncs.plotGeneral import rotate_xaxis


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((24, 10), (4, 5))
    subplotLabel(ax)

    X = read_h5ad("/opt/northwest_bal/full_fitted.h5ad")

    X = add_obs(X, "binary_outcome")
    X = add_obs(X, "patient_category")

    genes = bot_top_genes(X, cmp=1, geneAmount=6)

    for i, gene in enumerate(np.ravel(genes)):
        plot_avegene_per_status(X, gene, ax[i])
        rotate_xaxis(ax[i])

    return f
