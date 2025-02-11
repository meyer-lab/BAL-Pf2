"""Figure A1d_g"""

import anndata
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotGeneral import rotate_xaxis
from ..data_import import add_obs, condition_factors_meta, combine_cell_types
from ..utilities import bal_combine_bo_covid, cell_count_perc_df


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((10, 10), (4, 4))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    combine_cell_types(X)

    plot_cell_count(X, ax[0])

    cond_fact_meta_df = condition_factors_meta(X)
    
    plot_sample_count(cond_fact_meta_df, ax[1], ax[2], combine_categories=True, include_control=False)
    plot_sample_count(cond_fact_meta_df, ax[3], ax[4], combine_categories=False, include_control=True)

    cond_fact_meta_df = cond_fact_meta_df.drop_duplicates(subset=["patient_id"])
    plot_sample_count(cond_fact_meta_df, ax[5], ax[6], combine_categories=True, include_control=False)
    plot_sample_count(cond_fact_meta_df, ax[7], ax[8], combine_categories=False, include_control=True)
        
    for i in [2, 4]:
        ax[i].set(ylabel="Sample Proportion")
    for i in [5, 7]:
        ax[i].set(ylabel="Patient Count")
    for i in [6, 8]:
        ax[i].set(ylabel="Patient Proportion")

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
            ax=ax[i+9],
        )
        rotate_xaxis(ax[i+9])


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


def plot_sample_count(
    df: pd.DataFrame,
    ax1: Axes,
    ax2: Axes,
    combine_categories=True,
    include_control=True,
):
    """Plots overall patients in each category."""
    if include_control is False:
            df = df[df["patient_category"] != "Non-Pneumonia Control"]
      
    if combine_categories is True:
        comparison_column = "Status"
    else:
        comparison_column = "Uncombined"
        
    dfCond = (
        df.groupby(comparison_column, observed=True).size().reset_index(name="Sample Count")
    )

    if combine_categories is True: 
        sns.barplot(data=dfCond, x=comparison_column, y="Sample Count", hue="Status", ax=ax1)
    else: 
        sns.barplot(data=dfCond, x=comparison_column, y="Sample Count", color="k", ax=ax1)
        rotate_xaxis(ax1)

    total = dfCond["Sample Count"].sum()
    dfCond["Sample Count"] = dfCond["Sample Count"] / total

    if combine_categories is True: 
        sns.barplot(data=dfCond, x=comparison_column, y="Sample Count", hue="Status", ax=ax2)
    else:
        sns.barplot(data=dfCond, x=comparison_column, y="Sample Count", color="k", ax=ax2)
        rotate_xaxis(ax2)
    

