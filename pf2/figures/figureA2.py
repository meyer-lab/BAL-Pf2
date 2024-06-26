"""Figure A2: Weighted projections for each component"""

from anndata import read_h5ad
from .common import subplotLabel, getSetup
from .commonFuncs.plotPaCMAP import plot_wp_pacmap


def makeFigure():
    ax, f = getSetup((40, 30), (4, 10))
    subplotLabel(ax)

    X = read_h5ad("/opt/northwest_bal/full_fitted.h5ad", backed="r")

    for i in range(1, 41):
        plot_wp_pacmap(X, i, ax[i - 1], cbarMax=0.3)

    return f
