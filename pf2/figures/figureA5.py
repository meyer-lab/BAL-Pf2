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
    ax, f = getSetup((30, 30), (6, 6))
    subplotLabel(ax)

    X = read_h5ad("/opt/northwest_bal/full_fitted.h5ad")

    X = add_obs(X, "binary_outcome")
    X = add_obs(X, "patient_category")

    # genes = bot_top_genes(X, cmp=1, geneAmount=6)
    
    #cmp3 and 26
    # rows include pf2, then DEG and then Cellxgene
    genes = ["KIF20A", "NEK2", "PLK1", "CDC20", "FAM111B", "RAD54L", "UHRF1", 
             "GRN", "CD68", "FTL", "CTSD", "S100A11", "CD3E", "CD3D", "IL32", "CORO1A", "CD2", "LCK",
             "MIF", "SRM", "RPLP0", "RPL35", "ARL6IP1", "CENPF", "UBE2C", "PTTG1", "TOP2A"]

    for i, gene in enumerate(np.ravel(genes)):
        plot_avegene_per_status(X, gene, ax[i])
        rotate_xaxis(ax[i])

    return f
