"""
Figure A4a_6a
"""

import anndata
from .common import subplotLabel, getSetup
from .commonFuncs.plotFactors import plot_gene_factors_defined
from .commonFuncs.plotGeneral import rotate_yaxis

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((12, 10), (1, 2))

    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
  
    cmp1 = 10; cmp2 = 14; cmp3 = 15; cmp4 = 1; cmp5 = 4
    plot_gene_factors_defined([cmp1, cmp2, cmp3, cmp4, cmp5], X, ax[0], geneAmount=5)

    cmp1 = 22; cmp2 = 62; cmp3 = 55; cmp4 = 67
    plot_gene_factors_defined([cmp1, cmp3, cmp2, cmp4], X, ax[1], geneAmount=5)
    
    rotate_yaxis(ax[0], 180)
    rotate_yaxis(ax[1], 180)


    return f

