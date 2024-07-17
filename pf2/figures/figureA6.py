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
from ..data_import import add_obs, condition_factors_meta
import pandas as pd
import numpy as np


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 8), (2, 2))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    
    plot_cell_count(X, ax[0])
    
    cond_fact_meta_df = condition_factors_meta(X)
    plot_sample_count(cond_fact_meta_df, ax[1])
    
    
    celltype_count_perc_df = cell_count_perc_df(X, celltype="cell_type")
    celltype = np.unique(celltype_count_perc_df["Cell Type"])
    sns.boxplot(
        data=celltype_count_perc_df,
        x="Cell Type",
        y="Cell Type Percentage",
        hue="Status",
        order=celltype,
        showfliers=False,
        ax=ax[2],
    )
    rotate_xaxis(ax[2])
    

    return f


def plot_cell_count(X: anndata.AnnData, ax: Axes, cond: str = "sample_id",
                    status1: str = "binary_outcome", status2: str = "patient_category"):
    """Plots overall cell count."""
    df = X.obs[[cond, status1, status2]].reset_index(drop=True)
    
    df = df.replace({status1: {0: "Lived", 
                                1: "Dec."}})

    df = df.replace({status2: {"Non-Pneumonia Control": "Non-COVID", 
                                "Other Pneumonia": "Non-COVID",
                                "Other Viral Pneumonia": "Non-COVID"}})
    df["Status"] = df[status1] + df[status2]
    dfCond = df.groupby([cond, "Status"], observed=True).size().reset_index(name="Cell Count")

    sns.barplot(data=dfCond, x="Status", y="Cell Count", hue="Status", ax=ax)
    rotate_xaxis(ax)


def plot_sample_count(df: pd.DataFrame, ax: Axes, cond: str = "Condition", 
                    status1: str = "binary_outcome", status2: str = "patient_category"):
    """Plots overall patients in each category."""
    df = df[[status1, status2]].reset_index(drop=True)
    
    df = df.replace({status1: {0: "Lived", 
                                1: "Dec."}})

    df = df.replace({status2: {"Non-Pneumonia Control": "Non-COVID", 
                                "Other Pneumonia": "Non-COVID",
                                "Other Viral Pneumonia": "Non-COVID"}})
    df["Status"] = df[status1] + df[status2]
    dfCond = df.groupby(["Status"], observed=True).size().reset_index(name="Sample Count")

    sns.barplot(data=dfCond, x="Status", y="Sample Count", hue="Status", ax=ax)
    rotate_xaxis(ax)
    
    
    
def cell_count_perc_df(X, celltype="Cell Type"):
    """Returns DF with cell counts and percentages for experiment"""

    grouping = [celltype, "sample_id", "binary_outcome", "patient_category"]

    df = X.obs[grouping].reset_index(drop=True)


    df = df.replace({'binary_outcome': {0: "Died", 
                                1: "Lived"}})

    df = df.replace({'patient_category': {"Non-Pneumonia Control": "Non-COVID", 
                                "Other Pneumonia": "Non-COVID",
                                "Other Viral Pneumonia": "Non-COVID"}})
    df["Status"] = df["binary_outcome"] + df["patient_category"]

    dfCond = (
        df.groupby(["sample_id"], observed=True).size().reset_index(name="Cell Count")
    )
    dfCellType = (
        df.groupby([celltype, "sample_id", "Status"]
, observed=True).size().reset_index(name="Cell Count")
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
