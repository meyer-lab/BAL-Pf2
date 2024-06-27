"""Figure BS_A1: Description"""

from anndata import read_h5ad
from .common import subplotLabel, getSetup
from ..tensor import correct_conditions


def makeFigure():
    ax, f = getSetup((8, 12), (2, 1))
    subplotLabel(ax)

    X = read_h5ad("/opt/northwest_bal/full_fitted.h5ad", backed="r")

    X.uns["Pf2_A"] = correct_conditions(X)  
    condition_factors = X.uns["Pf2_A"]
    

    return f
