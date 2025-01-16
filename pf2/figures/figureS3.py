"""Figure A3: Most positively/negatively weighted genes for each component"""

import anndata
from .common import getSetup
from .commonFuncs.plotFactors import plot_gene_factors_partial


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((20, 20), (10, 10))

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")

    for i, cmp in enumerate([2, 25, 19, 26, 34, 27, 37, 44, 8, 31]):
        plot_gene_factors_partial(cmp, X, ax[2 * i], geneAmount=10, top=True)
        plot_gene_factors_partial(cmp, X, ax[2 * i + 1], geneAmount=10, top=False)

    return f
