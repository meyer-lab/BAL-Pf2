"""Figure A3: XX"""

import anndata
from .common import getSetup
from .commonFuncs.plotFactors import plot_gene_factors_partial


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((14, 24), (10, 8))

    X = anndata.read_h5ad("/opt/andrew/bal_partial_fitted.h5ad")

    for i in range(X.uns["Pf2_A"].shape[1]):
        plot_gene_factors_partial(i + 1, X, ax[2 * i], geneAmount=10, top=True)
        plot_gene_factors_partial(i + 1, X, ax[2 * i + 1], geneAmount=10, top=False)

    return f
