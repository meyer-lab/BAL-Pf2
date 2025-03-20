"""Figure S2b"""

from anndata import read_h5ad
from .common import getSetup, subplotLabel
from .commonFuncs.plotPaCMAP import plot_wp_per_celltype
from ..data_import import add_obs



def makeFigure():
    ax, f = getSetup((10, 10), (3, 3))
    subplotLabel(ax)

    X = read_h5ad("/opt/northwest_bal/full_fitted.h5ad", backed="r")
    add_obs(X, "patient_category")
    X = X[X.obs["patient_category"] != "Non-Pneumonia Control"] 

    for i, cmp in enumerate([10, 14, 1, 4, 22, 62, 55, 67]):
        plot_wp_per_celltype(X, cmp, ax[i])


    return f