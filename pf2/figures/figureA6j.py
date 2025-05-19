"""Figure S1: Cell type abundance and distribution across patient statuses"""

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
    ax, f = getSetup((7, 7), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    combine_cell_types(X)
    
    X = X[~X.obs["patient_category"].isin(["Non-Pneumonia Control", "COVID-19"])]

    celltype = ["combined_cell_type"]
    types = ["Cell Count", "Cell Type Percentage"]
    
    axs=0
    for i, celltypes in enumerate(celltype):
        for j, type in enumerate(types):
            celltype_count_perc_df = cell_count_perc_df(X, celltype=celltypes, include_control=False)
            new_df = celltype_count_perc_df.loc[celltype_count_perc_df["Cell Type"] == "Other"].copy().reset_index(drop=True)
            new_df["Cell Type"] = new_df["Cell Type"].astype(str)
            final_df = new_df.reset_index(drop=True)
            sns.boxplot(
                data=final_df,
                x="Cell Type",
                y=type,
                hue="Status",
                # order=celltype,
                showfliers=False,
                ax=ax[axs],
            )
            rotate_xaxis(ax[axs])
            axs+=1
        
    # X = X[X.obs["combined_cell_type"].isin(["Other"])]
    # plot_cell_count(X, ax[0])

    # ax[3].remove()

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

