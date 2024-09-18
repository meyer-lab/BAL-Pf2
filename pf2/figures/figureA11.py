"""
Figure A11:
"""

import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.axes import Axes
import anndata
from .common import subplotLabel, getSetup
from ..figures.commonFuncs.plotGeneral import bal_combine_bo_covid, rotate_xaxis
from ..data_import import add_obs, combine_cell_types
from .commonFuncs.plotFactors import bot_top_genes


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((12, 12), (4, 4))

    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    combine_cell_types(X)

    cmp1 = 27
    cmp2 = 46
    pos1 = True
    pos2 = True
    threshold = 0.5
    X = add_cmp_both_label(X, cmp1, cmp2, pos1, pos2, top_perc=threshold)

    genes1 = bot_top_genes(X, cmp=cmp1, geneAmount=4)
    genes2 = bot_top_genes(X, cmp=cmp2, geneAmount=4)
    genes = np.concatenate([genes1, genes2])

    for i, gene in enumerate(genes):
        plot_avegene_cmps(X, gene, ax[i], cmp1, cmp2)
        rotate_xaxis(ax[i])

    return f


def add_cmp_both_label(
    X: anndata.AnnData, cmp1: int, cmp2: int, pos1=True, pos2=True, top_perc=1
):
    """Adds if cells in top/bot percentage"""
    wprojs = X.obsm["weighted_projections"]
    pos_neg = [pos1, pos2]

    for i, cmp in enumerate([cmp1, cmp2]):
        if i == 0:
            if pos_neg[i] is True:
                thres_value = 100 - top_perc
                threshold1 = np.percentile(wprojs, thres_value, axis=0)
                idx = wprojs[:, cmp - 1] > threshold1[cmp - 1]

            else:
                thres_value = top_perc
                threshold1 = np.percentile(wprojs, thres_value, axis=0)
                idx = wprojs[:, cmp - 1] < threshold1[cmp - 1]

        if i == 1:
            if pos_neg[i] is True:
                thres_value = 100 - top_perc
                threshold2 = np.percentile(wprojs, thres_value, axis=0)
                idx = wprojs[:, cmp - 1] > threshold2[cmp - 1]
            else:
                thres_value = top_perc
                threshold2 = np.percentile(wprojs, thres_value, axis=0)
                idx = wprojs[:, cmp - 1] < threshold1[cmp - 1]

        X.obs[f"Cmp{cmp}"] = idx

    if pos1 and pos2 is True:
        idx = (wprojs[:, cmp1 - 1] > threshold1[cmp1 - 1]) & (
            wprojs[:, cmp2 - 1] > threshold2[cmp2 - 1]
        )
    elif pos1 and pos2 is False:
        idx = (wprojs[:, cmp1 - 1] < threshold1[cmp1 - 1]) & (
            wprojs[:, cmp2 - 1] < threshold2[cmp2 - 1]
        )
    elif pos1 is True and pos2 is False:
        idx = (wprojs[:, cmp1 - 1] > threshold1[cmp1 - 1]) & (
            wprojs[:, cmp2 - 1] < threshold2[cmp2 - 1]
        )
    elif pos1 is False and pos2 is True:
        idx = (wprojs[:, cmp1 - 1] < threshold1[cmp1 - 1]) & (
            wprojs[:, cmp2 - 1] > threshold2[cmp2 - 1]
        )

    X.obs["Both"] = idx

    return X


def plot_avegene_cmps(
    X: anndata.AnnData,
    gene: str,
    ax: Axes,
    cmp1: int,
    cmp2: int,
):
    """Plots average gene expression across cell types"""
    genesV = X[:, gene]
    dataDF = genesV.to_df()
    condition = "sample_id"
    status1 = "binary_outcome"
    status2 = "patient_category"
    cellType = "combined_cell_type"

    dataDF = dataDF.subtract(genesV.var["means"].values)
    dataDF[status1] = genesV.obs[status1].values
    dataDF[status2] = genesV.obs[status2].values
    dataDF["Condition"] = genesV.obs[condition].values
    dataDF["Cell Type"] = genesV.obs[cellType].values

    dataDF[f"Cmp{cmp1}"] = genesV.obs[f"Cmp{cmp1}"].values
    dataDF[f"Cmp{cmp2}"] = genesV.obs[f"Cmp{cmp2}"].values

    dataDF.loc[
        ((dataDF[f"Cmp{cmp1}"] == True) & (dataDF[f"Cmp{cmp2}"] == False), "Label")
    ] = f"Cmp{cmp1}"
    dataDF.loc[
        (dataDF[f"Cmp{cmp1}"] == False) & (dataDF[f"Cmp{cmp2}"] == True), "Label"
    ] = f"Cmp{cmp2}"
    dataDF.loc[
        (dataDF[f"Cmp{cmp1}"] == True) & (dataDF[f"Cmp{cmp2}"] == True), "Label"
    ] = "Both"
    dataDF.loc[
        (dataDF[f"Cmp{cmp1}"] == False) & (dataDF[f"Cmp{cmp2}"] == False), "Label"
    ] = "NoLabel"

    dataDF = dataDF.dropna(subset="Label")
    dataDF = bal_combine_bo_covid(dataDF, status1, status2)

    df = pd.melt(
        dataDF, id_vars=["Label", "Condition", "Cell Type"], value_vars=gene
    ).rename(columns={"variable": "Gene", "value": "Value"})

    df = df.groupby(["Label", "Gene", "Condition", "Cell Type"], observed=False).mean()
    df = df.rename(columns={"Value": "Average Gene Expression"}).reset_index()

    sns.boxplot(
        data=df.loc[df["Gene"] == gene],
        x="Label",
        y="Average Gene Expression",
        hue="Cell Type",
        ax=ax,
        showfliers=False,
    )
    ax.set(ylabel=f"Average {gene}")

    return df
