"""Figure S2b"""

from anndata import read_h5ad
from .common import getSetup, subplotLabel
from .commonFuncs.plotPaCMAP import plot_wp_per_celltype
from ..data_import import add_obs



def makeFigure():
    ax, f = getSetup((50, 50), (20, 20))
    # subplotLabel(ax)

    X = read_h5ad("/opt/northwest_bal/full_fitted.h5ad", backed="r")
    add_obs(X, "patient_category")
    X = X[X.obs["patient_category"] != "Non-Pneumonia Control"] 

    for i in range(80):
        plot_wp_per_celltype(X, i+1, ax[i])


    return f