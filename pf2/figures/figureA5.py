"""Figure A5: Plots average gene expression by status and cell type for a component"""

from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
from .commonFuncs.plotFactors import bot_top_genes
from .commonFuncs.plotGeneral import plot_avegene_per_status
from pf2.data_import import add_obs
from pf2.figures.commonFuncs.plotGeneral import rotate_xaxis


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((24, 10), (4, 5))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    print(X)
    print(len(np.unique(X.obs["sample_id"])))
    # X = add_obs(X, "binary_outcome")
    # X = add_obs(X, "patient_category")
    


    # genes = bot_top_genes(X, cmp=42, geneAmount=10)
    
    # print("BOT GENES")
    # for i in genes[:30]: 
    #     print(i)
        
    # print("TOP GENES")
    
    # for i in genes[30:]: 
    #     print(i)
        


    # for i, gene in enumerate(np.ravel(genes)):
    #     plot_avegene_per_status(X, gene, ax[i])
    #     rotate_xaxis(ax[i])

    return f
