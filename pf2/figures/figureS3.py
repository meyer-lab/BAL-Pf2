"""Figure S3"""

import anndata
from .common import getSetup
from .commonFuncs.plotFactors import plot_gene_factors_partial


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((12, 12), (5, 5))

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")

    for i, cmp in enumerate([3, 26, 20, 27, 13, 28, 38, 45, 9, 32]):
        plot_gene_factors_partial(cmp, X, ax[2 * i], geneAmount=10, top=True)
        plot_gene_factors_partial(cmp, X, ax[2 * i + 1], geneAmount=10, top=False)

    return f
