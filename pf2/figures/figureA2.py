"""Figure A2: Weighted projections for each component"""

from anndata import read_h5ad
from .common import getSetup
from .commonFuncs.plotPaCMAP import plot_wp_pacmap, plot_wp_per_celltype


def makeFigure():
    ax, f = getSetup((8, 8), (2, 2))
    # subplotLabel(ax)

    X = read_h5ad("/opt/northwest_bal/full_fitted.h5ad", backed="r")

    for i, cmp in enumerate([15, 16, 19]):
        plot_wp_pacmap(X, cmp, ax[i - 1], cbarMax=0.4)
        # plot_wp_per_celltype(X, i, ax[i-1])

    return f
