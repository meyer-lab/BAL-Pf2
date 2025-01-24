"""Figure S8"""

from anndata import read_h5ad
from .common import getSetup
from .commonFuncs.plotPaCMAP import plot_wp_pacmap, plot_wp_per_celltype
from matplotlib.axes import Axes
import seaborn as sns
import pandas as pd
import numpy as np


def makeFigure():
    ax, f = getSetup((20, 20), (5, 5))
    # subplotLabel(ax)

    X = read_h5ad("/opt/northwest_bal/full_fitted.h5ad", backed="r")

    plot_pair_cond_factors(X, 20, 27, ax[0], "Label")
        

    return f




def plot_pair_cond_factors(
    X, cmp1: int, cmp2: int, ax: Axes, label: str
):
    """Plots two condition components weights"""
    factors = np.array(X.uns["Pf2_A"])
    XX = factors
    factors -= np.median(XX, axis=0)
    factors /= np.std(XX, axis=0)
    
    df = pd.DataFrame(factors, columns=[f"Cmp. {i}" for i in range(1, factors.shape[1] + 1)])

    sns.scatterplot(data=df, x=f"Cmp. {cmp1}", y=f"Cmp. {cmp2}", ax=ax)
    ax.set(title="Condition Factors")
    
    
