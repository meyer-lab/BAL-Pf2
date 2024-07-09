"""
Lupus: Cell type percentage between status (with stats comparison) and
correlation between component and cell count/percentage for each cell type
"""
import anndata
from pf2.figures.common import getSetup, subplotLabel
from pf2.tensor import correct_conditions

from pf2.data_import import  add_obs
from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr
from .commonFuncs.plotGeneral import rotate_xaxis
from matplotlib.axes import Axes
import anndata


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((30, 30), (5, 10))
    ax, f = getSetup((12, 10), (2, 2))
    

    # Add subplot labels
    subplotLabel(ax)
    
    X = read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    X = add_obs(X, "binary_outcome")
    X = add_obs(X, "patient_category")
    X.uns["Pf2_A"] = correct_conditions(X)

    celltype_count_perc_df = cell_count_perc_df(X, celltype="cell_type")
    celltype = np.unique(celltype_count_perc_df["Cell Type"])
    sns.boxplot(
        data=celltype_count_perc_df,
        x="Cell Type",
        y="Cell Type Percentage",
        hue="binary_outcome",
        order=celltype,
        showfliers=False,
        ax=ax[0],
    )
    rotate_xaxis(ax[0])

    # for i in range(1):
    #     idx = len(np.unique(celltype_count_perc_df["Cell Type"]))
    #     plot_correlation_cmp_cell_count_perc(
    #         X, i+1, celltype_count_perc_df, ax[i+1], cellPerc=True
    #     )
    #     print(i)

    return f


def cell_count_perc_df(X, celltype="Cell Type"):
    """Returns DF with cell counts and percentages for experiment"""

    grouping = [celltype, "sample_id", "binary_outcome", "patient_category"]

    df = X.obs[grouping].reset_index(drop=True)

    
    df = df.replace({'binary_outcome': {0: "Died", 
                                1: "Lived"}})
    
    df = df.replace({'patient_category': {"Non-Pneumonia Control": "Non-COVID", 
                                "Other Pneumonia": "Non-COVID",
                                "Other Viral Pneumonia": "Non-COVID"}})
    df["binary_outcome"] = df["binary_outcome"] + df["patient_category"]

    dfCond = (
        df.groupby(["sample_id"], observed=True).size().reset_index(name="Cell Count")
    )
    dfCellType = (
        df.groupby(grouping, observed=True).size().reset_index(name="Cell Count")
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


def plot_correlation_cmp_cell_count_perc(
    X: anndata, cmp: int, cellcountDF: pd.DataFrame, ax: Axes, cellPerc=True
):
    """Plot component weights by cell type count or percentage for a cell type"""
    yt = np.unique(X.obs["sample_id"])
    factorsA = np.array(X.uns["Pf2_A"])
    factorsA = factorsA[:, cmp - 1]
    if cellPerc is True:
        cellPerc = "Cell Type Percentage"
    else:
        cellPerc = "Cell Count"
    totaldf = pd.DataFrame([])
    correlationdf = pd.DataFrame([])
    cellcountDF["sample_id"] = pd.Categorical(cellcountDF["sample_id"], yt)
    for i, celltype in enumerate(np.unique(cellcountDF["Cell Type"])):
        for j, cond in enumerate(np.unique(cellcountDF["sample_id"])):
            status = np.unique(
                cellcountDF.loc[cellcountDF["sample_id"] == cond]["binary_outcome"]
            )
            smalldf = cellcountDF.loc[
                (cellcountDF["sample_id"] == cond)
                & (cellcountDF["Cell Type"] == celltype)
            ]
            if smalldf.empty is False:
                smalldf = smalldf.assign(Cmp=factorsA[j])
            else:
                smalldf = pd.DataFrame(
                    {
                        "sample_id": cond,
                        "Cell Type": celltype,
                        "binary_outcome": status,
                        cellPerc: 0,
                        "Cmp": factorsA[j],
                    }
                )

            totaldf = pd.concat([totaldf, smalldf])

        df = totaldf.loc[totaldf["Cell Type"] == celltype]
        
        print("Cell Type:", celltype)
        print("Cell Count:", df["Cell Count"].mean())
        print("Cell Type Percentage:", df["Cell Type Percentage"].mean())
        pearson = pearsonr(df["Cmp"], df[cellPerc])[0]

        correl = [pearson]
        test = ["Pearson"]

        for k in range(1):
            correlationdf = pd.concat(
                [
                    correlationdf,
                    pd.DataFrame(
                        {
                            "Cell Type": celltype,
                            "Correlation": [test[k]],
                            "Value": [correl[k]],
                        }
                    ),
                ]
            )

    sns.barplot(
        data=correlationdf, x="Cell Type", y="Value", hue="Correlation", ax=ax
    )
    rotate_xaxis(ax)
    ax.set(title=f"Cmp. {cmp} V. {cellPerc}")