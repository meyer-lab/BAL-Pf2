"""Figure S1"""

import anndata
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotGeneral import rotate_xaxis
from ..data_import import add_obs, combine_cell_types
from ..utilities import bal_combine_bo_covid, cell_count_perc_df


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((14, 6), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    combine_cell_types(X)

    plot_cell_count(X, ax[0])

    celltype = ["cell_type", "combined_cell_type"]
    for i, celltypes in enumerate(celltype):
        celltype_count_perc_df = cell_count_perc_df(X, celltype=celltypes, include_control=False)
        celltype = np.unique(celltype_count_perc_df["Cell Type"])
        sns.boxplot(
            data=celltype_count_perc_df,
            x="Cell Type",
            y="Cell Type Percentage",
            hue="Status",
            order=celltype,
            showfliers=False,
            ax=ax[i+1],
        )
        rotate_xaxis(ax[i+1])

    ax[3].remove()

    return f


def plot_cell_count(
    X: anndata.AnnData,
    ax: Axes,
    cond: str = "sample_id",
    status1: str = "binary_outcome",
    status2: str = "patient_category",
):
    """Plots overall cell count."""
    df = X.obs[[cond, status1, status2]].reset_index(drop=True)

    df = bal_combine_bo_covid(df)
    dfCond = (
        df.groupby([cond, "Status"], observed=True)
        .size()
        .reset_index(name="Cell Count")
    )

    sns.barplot(data=dfCond, x="Status", y="Cell Count", hue="Status", ax=ax)
    rotate_xaxis(ax)

