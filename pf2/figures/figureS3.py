"""Figure S3"""

import anndata
from .common import getSetup
from .commonFuncs.plotFactors import plot_gene_factors_partial


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((10, 10), (4, 4))

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")

    for i, cmp in enumerate([10, 14, 16, 34, 23, 15, 55, 67]):
        plot_gene_factors_partial(cmp, X, ax[2 * i], geneAmount=10, top=True)
        plot_gene_factors_partial(cmp, X, ax[2 * i + 1], geneAmount=10, top=False)

    return f
#    10
# 13 -0.229122         14
# 15 -0.228099         16
# 33 -0.211750         34
# 22 -0.210493         23