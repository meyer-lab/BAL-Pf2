"""Figure S2a"""

from anndata import read_h5ad
from .common import getSetup, subplotLabel
from .commonFuncs.plotPaCMAP import plot_wp_pacmap
from ..data_import import add_obs



def makeFigure():
    ax, f = getSetup((12, 12), (4, 4))
    subplotLabel(ax)

    X = read_h5ad("/opt/northwest_bal/full_fitted.h5ad", backed="r")
    add_obs(X, "patient_category")
    X = X[X.obs["patient_category"] != "Non-Pneumonia Control"] 

    for i, cmp in enumerate([3, 10, 14, 15, 16, 23, 34, 55, 67, 22, 62, 1, 4]):
        plot_wp_pacmap(X, cmp, ax[i], cbarMax=0.4)
        
    for i in [13, 14, 15]:
        ax[i].remove()
        

    return f
