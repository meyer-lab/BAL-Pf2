"""Figure A1: XX"""

# import time
# from pf2.data_import import import_data
# from pf2.tensor import pf2
import anndata
from pf2.figures.common import getSetup, subplotLabel
from pf2.tensor import correct_conditions
from pf2.figures.commonFuncs.plotFactors import (
    plot_condition_factors,
    plot_gene_factors,
    plot_eigenstate_factors,
)
from pf2.figures.commonFuncs.plotPaCMAP import plot_labels_pacmap


def makeFigure():
    ax, f = getSetup((18, 6), (1, 4))
    subplotLabel(ax)

    # start = time.time()
    # data = import_data()
    # data = sc.pp.subsample(data, fraction=.4, random_state=1, copy=True)
    # X, _ = pf2(data, rank=40)
    # print(f"Factorization Time: {time.time() - start} sec")
    # X.write("bal_fitted.h5ad")

    X = anndata.read_h5ad("/opt/andrew/bal_partial_fitted.h5ad")
    X.uns["Pf2_A"] = correct_conditions(X)
    plot_condition_factors(X, ax[0], cond="batch")
    plot_eigenstate_factors(X, ax[1])
    plot_gene_factors(X, ax[2])
    plot_labels_pacmap(X, "cell_type", ax[3])

    return f
