"""Figure S2"""

from anndata import read_h5ad
from .common import getSetup
from .commonFuncs.plotPaCMAP import plot_wp_pacmap, plot_wp_per_celltype


def makeFigure():
    ax, f = getSetup((20, 20), (5, 5))
    # subplotLabel(ax)

    X = read_h5ad("/opt/northwest_bal/full_fitted.h5ad", backed="r")
    add_obs(X, "patient_category")
    X = X[X.obs["patient_category"] != "Non-Pneumonia Control"] 

    for i, cmp in enumerate([3, 26, 20, 27, 35, 28, 38, 45, 9, 32]):
        plot_wp_pacmap(X, cmp, ax[i], cbarMax=0.4)
        plot_wp_per_celltype(X, cmp, ax[i+10])
        

    return f
