"""Figure BS_A1: Description"""

import anndata
from .common import subplotLabel, getSetup
from ..data_import import condition_factors_meta
from ..tensor import correct_conditions


def makeFigure():
    ax, f = getSetup((8, 12), (2, 1))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    X.uns["Pf2_A"] = correct_conditions(X)
    cond_factors_df = condition_factors_meta(X)

    return f
