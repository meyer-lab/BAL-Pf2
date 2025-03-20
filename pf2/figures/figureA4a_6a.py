"""
Figure A4a_6a
"""

import anndata
from .common import subplotLabel, getSetup
from .commonFuncs.plotFactors import plot_gene_factors_defined

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((12, 10), (1, 2))

    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
  
    cmp1 = 3; cmp2 = 13; cmp3 = 20; cmp4 = 26; cmp5 = 27
    plot_gene_factors_defined([cmp1, cmp2, cmp3, cmp4, cmp5], X, ax[0], geneAmount=5)

    cmp1 = 9; cmp2 = 32; cmp3 = 28; cmp4 = 38; cmp5 = 45
    plot_gene_factors_defined([cmp1, cmp3, cmp2, cmp4, cmp5], X, ax[1], geneAmount=5)


    return f

