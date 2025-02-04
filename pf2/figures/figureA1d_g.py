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
from .commonFuncs.plotGeneral import rotate_xaxis, bal_combine_bo_covid
from ..data_import import add_obs, condition_factors_meta, combine_cell_types


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((10, 10), (4, 4))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")

    plot_cell_count(X, ax[0])

    cond_fact_meta_df = condition_factors_meta(X)
    
    
    plot_sample_count(cond_fact_meta_df, ax[1], ax[2], combine=True, include_control=False, sample_id=True)
    plot_sample_count(cond_fact_meta_df, ax[3], ax[4], combine=False, sample_id=True)
    
    cond_fact_meta_df = cond_fact_meta_df.drop_duplicates(subset=["patient_id"])
    plot_sample_count(cond_fact_meta_df, ax[5], ax[6], combine=True, include_control=False, sample_id=False)
    plot_sample_count(cond_fact_meta_df, ax[7], ax[8], combine=False, sample_id=False)
    
    X = X[X.obs["patient_category"] != "Non-Pneumonia Control"] 

    celltype_count_perc_df = cell_count_perc_df(X, celltype="cell_type")
    celltype = np.unique(celltype_count_perc_df["Cell Type"])
    sns.boxplot(
        data=celltype_count_perc_df,
        x="Cell Type",
        y="Cell Type Percentage",
        hue="Status",
        order=celltype,
        showfliers=False,
        ax=ax[9],
    )
    rotate_xaxis(ax[9])

    combine_cell_types(X)
    celltype_count_perc_df = cell_count_perc_df(X, celltype="combined_cell_type")
    celltype = np.unique(celltype_count_perc_df["Cell Type"])
    sns.boxplot(
        data=celltype_count_perc_df,
        x="Cell Type",
        y="Cell Type Percentage",
        hue="Status",
        order=celltype,
        showfliers=False,
        ax=ax[10],
    )
    rotate_xaxis(ax[10])

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
    status1: str = "binary_outcome",
    status2: str = "patient_category",
    combine=True,
    include_control=True,
    sample_id=True
):
    """Plots overall patients in each category."""
    df = df[[status1, status2]].reset_index(drop=True)

    if combine is True:
        if include_control is False:
            df = df[df[status2] != "Non-Pneumonia Control"]
        df = bal_combine_bo_covid(df)

    else:
        df = df.replace({status1: {0: "L-", 1: "D-."}})
        df["Status"] = df[status1] + df[status2]

    dfCond = (
        df.groupby(["Status"], observed=True).size().reset_index(name="Sample Count")
    )

    if combine is True: 
        sns.barplot(data=dfCond, x="Status", y="Sample Count", hue="Status", ax=ax1)
    else: 
        sns.barplot(data=dfCond, x="Status", y="Sample Count", color="k", ax=ax1)
        rotate_xaxis(ax1)

    if sample_id is False:
        ax1.set(ylabel="Patient Count")
        
    total = dfCond["Sample Count"].sum()
    dfCond["Sample Count"] = dfCond["Sample Count"] / total

    if combine is True: 
        sns.barplot(data=dfCond, x="Status", y="Sample Count", hue="Status", ax=ax2)
    else:
        sns.barplot(data=dfCond, x="Status", y="Sample Count", color="k", ax=ax2)
        rotate_xaxis(ax2)
    
    if sample_id is False:
        ax2.set(ylabel="Patient Proportion")
    else:
        ax2.set(ylabel="Sample Proportion")


def cell_count_perc_df(X, celltype="Cell Type", include_control=True):
    """Returns DF with cell counts and percentages for experiment"""

    grouping = [celltype, "sample_id", "binary_outcome", "patient_category"]
    df = X.obs[grouping].reset_index(drop=True)
    if include_control is False:
         df = df[df["patient_category"] != "Non-Pneumonia Control"]
    df = bal_combine_bo_covid(df)

    dfCond = (
        df.groupby(["sample_id"], observed=True).size().reset_index(name="Cell Count")
    )
    dfCellType = (
        df.groupby([celltype, "sample_id", "Status"], observed=True)
        .size()
        .reset_index(name="Cell Count")
    )
    dfCellType["Cell Count"] = dfCellType["Cell Count"].astype("float")

    dfCellType["Cell Type Percentage"] = 0.0
    for cond in np.unique(df["sample_id"]):
        dfCellType.loc[dfCellType["sample_id"] == cond, "Cell Type Percentage"] = (
            100
            * dfCellType.loc[dfCellType["sample_id"] == cond, "Cell Count"].to_numpy()
            / dfCond.loc[dfCond["sample_id"] == cond]["Cell Count"].to_numpy()
        )

    dfCellType.rename(columns={celltype: "Cell Type"}, inplace=True)

    return dfCellType
