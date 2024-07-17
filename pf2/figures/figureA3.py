"""Figure A3: Most positively/negatively weighted genes for each component"""

import anndata
from .common import getSetup, subplotLabel
from .commonFuncs.plotFactors import plot_gene_factors_partial


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((20, 20), (10, 10))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")

    for i in range(X.uns["Pf2_A"].shape[1]):
        plot_gene_factors_partial(i + 1, X, ax[2 * i], geneAmount=10, top=True)
        plot_gene_factors_partial(i + 1, X, ax[2 * i + 1], geneAmount=10, top=False)

    return f
