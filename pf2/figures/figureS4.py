"""Figure S4: Top/bottom gene loadings for selected components"""

import anndata
from .common import getSetup
from .commonFuncs.plotFactors import plot_gene_factors_partial


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((12, 8), (5, 6))

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")

    for i, cmp in enumerate([3, 10, 14, 15, 16, 23, 34, 55, 67, 22, 62, 1, 4]):
        plot_gene_factors_partial(cmp, X, ax[2 * i], geneAmount=10, top=True)
        plot_gene_factors_partial(cmp, X, ax[2 * i + 1], geneAmount=10, top=False)
        
        
    for i in [26, 27, 28, 29]:
        ax[i].remove()
        

    return f
