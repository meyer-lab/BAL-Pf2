"""
Figure S8
"""

import anndata
from .common import subplotLabel, getSetup
from .commonFuncs.plotPaCMAP import plot_gene_pacmap
from ..data_import import add_obs

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((10, 10), (4, 4))

    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "patient_category")
    X = X[X.obs["patient_category"] != "Non-Pneumonia Control"] 

    genes = ["CCNO", "FOXJ1", "TPPP3", "MUC5AC", "MUC5B", "SCGB1A1", "SCGB3A1"]

    for i, gene in enumerate(genes):
        plot_gene_pacmap(gene, X, ax[i+1])
    
  

    return f


