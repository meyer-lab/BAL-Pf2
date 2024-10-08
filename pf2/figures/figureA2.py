"""Figure A2: Weighted projections for each component"""

from anndata import read_h5ad
from .common import getSetup
from .commonFuncs.plotPaCMAP import plot_wp_pacmap, plot_wp_per_celltype


def makeFigure():
    ax, f = getSetup((40, 30), (5, 10))
    # subplotLabel(ax)

    X = read_h5ad("/opt/northwest_bal/full_fitted.h5ad", backed="r")

    for i in [24, 25, 49]:
        plot_wp_pacmap(X, i, ax[i - 1], cbarMax=0.4)
        # plot_wp_per_celltype(X, i, ax[i-1])

    return f
