"""
Figure 6
"""

import anndata
from .common import subplotLabel, getSetup
from .commonFuncs.plotFactors import plot_gene_factors_defined
from .commonFuncs.plotGeneral import rotate_xaxis

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((12, 10), (1, 4))

    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
  
    cmp1 = 3; cmp2 = 20; cmp3 = 26; cmp4 = 27; cmp5 = 35
    plot_gene_factors_defined([cmp1, cmp3, cmp2, cmp4, cmp5], X, ax[0], geneAmount=5)

    cmp1 = 9; cmp2 = 32; cmp3 = 28; cmp4 = 38; cmp5 = 45
    plot_gene_factors_defined([cmp1, cmp3, cmp2, cmp4, cmp5], X, ax[1], geneAmount=5)

    # cmp1 = 9; cmp2 = 32
    # plot_gene_factors_defined([cmp1, cmp2], X, ax[2], geneAmount=5)

    # cmp1 = 28; cmp2 = 38; cmp3 = 45
    # plot_gene_factors_defined([cmp1, cmp2, cmp3], X, ax[3], geneAmount=5)

  

    return f

