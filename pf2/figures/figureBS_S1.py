"""Figure BS_S1: Description"""

import anndata
from .common import subplotLabel, getSetup
from ..tensor import correct_conditions


def makeFigure():
    ax, f = getSetup((8, 12), (2, 1))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")

    X.uns["Pf2_A"] = correct_conditions(X)  
    condition_factors = X.uns["Pf2_A"]
    

    return f
