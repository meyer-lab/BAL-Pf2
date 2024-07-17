import numpy as np
import pandas as pd
import seaborn as sns
import scanpy as sc
import anndata
from matplotlib.axes import Axes


def plot_avegene_per_status(
    X: anndata.AnnData,
    gene: str,
    ax: Axes,
    condition="sample_id",
    cellType="cell_type",
    status="binary_outcome",
    status2="patient_category"
):
    """Plots average gene expression across cell types for a category of drugs"""
    genesV = X[:, gene]
    dataDF = genesV.to_df()
    dataDF = dataDF.subtract(genesV.var["means"].values)
    dataDF["Status"] = genesV.obs[status].values
    dataDF["Condition"] = genesV.obs[condition].values
    dataDF["Cell Type"] = genesV.obs[cellType].values

    df = pd.melt(
        dataDF, id_vars=["Status", "Cell Type", "Condition"], value_vars=gene
    ).rename(columns={"variable": "Gene", "value": "Value"})

    df = df.groupby(["Status", "Cell Type", "Gene", "Condition"], observed=False).mean()
    df = df.rename(columns={"Value": "Average Gene Expression"}).reset_index()
    
    dataDF = dataDF.replace({'Status': {0: "Lived", 
                                1: "Dec."}})

    dataDF["Status2"] = genesV.obs[status2].values
    dataDF = dataDF.replace({'Status2': {"Non-Pneumonia Control": "Non-COVID", 
                                "Other Pneumonia": "Non-COVID",
                                "Other Viral Pneumonia": "Non-COVID"}})
    dataDF["Status"] = dataDF["Status2"] + dataDF["Status"]

    sns.boxplot(
        data=df.loc[df["Gene"] == gene],
        x="Cell Type",
        y="Average Gene Expression",
        hue="Status",
        ax=ax,
        showfliers=False,
    )
    ax.set(ylabel=f"Average {gene}")

    return df


def rotate_xaxis(ax, rotation=90):
    """Rotates text by 90 degrees for x-axis"""
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=rotation)


def rotate_yaxis(ax, rotation=90):
    """Rotates text by 90 degrees for y-axis"""
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(labels=ax.get_yticklabels(), rotation=rotation)
