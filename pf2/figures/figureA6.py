"""Figure A6: XX"""

from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
import seaborn as sns
from matplotlib.axes import Axes
import anndata


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((24, 10), (1, 1))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/bal_partial_fitted.h5ad")

    plot_cell_count(X, ax[0])

    return f


def rotate_xaxis(ax, rotation=90):
    """Rotates text by 90 degrees for x-axis"""
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=rotation)


def plot_cell_count(X: anndata.AnnData, ax: Axes, cond: str = "batch"):
    """Plots overall cell count for Chen et al."""
    df = X.obs[[cond]].reset_index(drop=True)
    dfCond = df.groupby([cond], observed=True).size().reset_index(name="Cell Count")

    sns.barplot(data=dfCond, x=cond, y="Cell Count", color="k", ax=ax)
    rotate_xaxis(ax)
