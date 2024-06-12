"""Figure A2: Weighted projections for each component"""

import anndata
from .common import subplotLabel, getSetup
from .commonFuncs.plotPaCMAP import plot_wp_pacmap
# from .commonFuncs.plotPaCMAP import plot_wp_per_celltype


def makeFigure():
    ax, f = getSetup((40, 30), (4, 10))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/partial_fitted.h5ad")

    for i in range(1, 41):
        plot_wp_pacmap(X, i, ax[i - 1], cbarMax=0.3)
        # plot_wp_per_celltype(X, i, ax[i-1])

    return f
