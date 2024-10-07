"""
Figure A7:
"""

import anndata
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr
from matplotlib.axes import Axes
from ..figures.common import getSetup
from ..tensor import correct_conditions
from ..data_import import add_obs
from .commonFuncs.plotGeneral import rotate_xaxis
from ..figures.figureA6 import cell_count_perc_df


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((10, 10), (2, 2))

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    X = add_obs(X, "binary_outcome")
    X = add_obs(X, "patient_category")
    X.uns["Pf2_A"] = correct_conditions(X)

    celltype_count_perc_df = cell_count_perc_df(X, celltype="cell_type")

    for i, cmp in enumerate([27, 46, 23, 34]):
        plot_correlation_cmp_cell_count_perc(
            X, cmp, celltype_count_perc_df, ax[i], cellPerc=True
        )

    return f


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
                cellcountDF.loc[cellcountDF["sample_id"] == cond]["Status"]
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
                        "Status": status,
                        cellPerc: 0,
                        "Cmp": factorsA[j],
                    }
                )

            totaldf = pd.concat([totaldf, smalldf])

        df = totaldf.loc[totaldf["Cell Type"] == celltype]
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

    sns.barplot(data=correlationdf, x="Cell Type", y="Value", hue="Correlation", ax=ax)
    rotate_xaxis(ax)
    ax.set(title=f"Cmp. {cmp} V. {cellPerc}")
