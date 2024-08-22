import pandas as pd
import seaborn as sns
import anndata
from matplotlib.axes import Axes


def plot_avegene_per_status(
    X: anndata.AnnData,
    gene: str,
    ax: Axes,
    condition="sample_id",
    cellType="cell_type",
    status1="binary_outcome",
    status2="patient_category"
):
    """Plots average gene expression across cell types for a category of drugs"""
    genesV = X[:, gene]
    dataDF = genesV.to_df()
    dataDF = dataDF.subtract(genesV.var["means"].values)
    dataDF[status1] = genesV.obs[status1].values
    dataDF[status2] = genesV.obs[status2].values
    dataDF["Condition"] = genesV.obs[condition].values
    dataDF["Cell Type"] = genesV.obs[cellType].values
    
    df = bal_combine_bo_covid(dataDF, status1, status2)

    df = pd.melt(
        df, id_vars=["Status", "Cell Type", "Condition"], value_vars=gene
    ).rename(columns={"variable": "Gene", "value": "Value"})

    df = df.groupby(["Status", "Cell Type", "Gene", "Condition"], observed=False).mean()
    df = df.rename(columns={"Value": "Average Gene Expression"}).reset_index()
    

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

def bal_combine_bo_covid(df, status1: str = "binary_outcome", status2: str = "patient_category"):
    """Combines binary outcome and covid status columns"""
    df = df.replace({status1: {0: "L-", 
                                1: "D-"}})

    df = df.replace({status2: {"COVID-19": "C19",
                                "Non-Pneumonia Control": "nC19", 
                                "Other Pneumonia": "nC19",
                                "Other Viral Pneumonia": "nC19"}})
    df["Status"] = df[status1] + df[status2]

    return df


def rotate_xaxis(ax, rotation=90):
    """Rotates text by 90 degrees for x-axis"""
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=rotation)


def rotate_yaxis(ax, rotation=90):
    """Rotates text by 90 degrees for y-axis"""
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(labels=ax.get_yticklabels(), rotation=rotation)
