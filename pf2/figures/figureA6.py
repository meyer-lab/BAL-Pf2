"""Figure A6: Plots cell count per patient"""

from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
import seaborn as sns
from matplotlib.axes import Axes
import anndata
from pf2.figures.commonFuncs.plotGeneral import rotate_xaxis


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((24, 10), (1, 1))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/northwest_bal/full_fitted.h5ad")

    plot_cell_count(X, ax[0])

    return f


def plot_cell_count(X: anndata.AnnData, ax: Axes, cond: str = "sample_id"):
    """Plots overall cell count for Chen et al."""
    df = X.obs[[cond]].reset_index(drop=True)
    dfCond = df.groupby([cond], observed=True).size().reset_index(name="Cell Count")

    print(dfCond.sort_values("Cell Count"))
    sns.barplot(data=dfCond, x=cond, y="Cell Count", color="k", ax=ax)
    rotate_xaxis(ax)
