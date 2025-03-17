import pandas as pd
import seaborn as sns
import anndata
from matplotlib.axes import Axes
import numpy as np
from ...utilities import bal_combine_bo_covid, cell_count_perc_df, move_index_to_column, aggregate_anndata
from ...predict import predict_mortality_all

def plot_avegene_per_status(
    X: anndata.AnnData,
    gene: str,
    ax: Axes,
    condition="sample_id",
    cellType="cell_type",
    status1="binary_outcome",
    status2="patient_category",
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


def plot_avegene_cmps(
    X: anndata.AnnData,
    gene: str,
    ax: Axes,
    order=None
):
    """Plots average gene expression across cell types"""
    genesV = X[:, gene]
    dataDF = genesV.to_df()
    condition = "sample_id"
    status1 = "binary_outcome"
    status2 = "patient_category"
    cellType = "combined_cell_type"

    dataDF[status1] = genesV.obs[status1].values
    dataDF[status2] = genesV.obs[status2].values
    dataDF["Condition"] = genesV.obs[condition].values
    dataDF["Cell Type"] = genesV.obs[cellType].values
    dataDF["Label"] = genesV.obs["Label"].values
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
        ax=ax,
        order=order,
        showfliers=False,
    )
    ax.set(ylabel=f"Average {gene}")

    return df


def plot_pair_gene_factors(X: anndata.AnnData, cmp1: int, cmp2: int, ax: Axes):
    """Plots two gene components weights"""
    cmpWeights = np.concatenate(
        ([X.varm["Pf2_C"][:, cmp1 - 1]], [X.varm["Pf2_C"][:, cmp2 - 1]])
    )
    df = pd.DataFrame(
        data=cmpWeights.transpose(), columns=[f"Cmp. {cmp1}", f"Cmp. {cmp2}"]
    )
    sns.scatterplot(data=df, x=f"Cmp. {cmp1}", y=f"Cmp. {cmp2}", ax=ax, color="k")
    ax.set(title="Gene Factors")


def plot_toppfun(cmp, ax):
    """Plot GSEA results"""
    df = pd.read_csv(f"pf2/data/topp_fun_cmp{cmp}.csv", dtype=str)
    df = df.drop(columns=["ID", "Verbose ID"])
    category = df["Category"].to_numpy().astype(str)

    df = df.drop(columns=["Category"])
    df["Process"] = category
    df = df.iloc[:1000, :]
    df["Total Genes"] = df.iloc[:, 2:-1].astype(int).sum(axis=1).to_numpy()
    df = df.loc[df.loc[:, "Process"] == "GO: Biological Process"]
    df["pValue"] = df["pValue"].astype(float)

    sns.scatterplot(
        data=df.iloc[:10, :], x="pValue", y="Name", hue="Total Genes", ax=ax
    )
    ax.set(xscale="log")


def rotate_xaxis(ax, rotation=90):
    """Rotates text by 90 degrees for x-axis"""
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=rotation)


def rotate_yaxis(ax, rotation=90):
    """Rotates text by 90 degrees for y-axis"""
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(labels=ax.get_yticklabels(), rotation=rotation)


def plot_correlation_heatmap(correlation_df: pd.DataFrame, xticks, yticks, ax: Axes, mask=None):
    """Plots a heatmap of the correlation matrix"""
    cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)
    
    if mask is not None: 
        mask = np.triu(np.ones_like(correlation_df, dtype=bool))
        for i in range(len(mask)):
            mask[i, i] = False
        
    sns.heatmap(
        data=correlation_df.to_numpy(),
        vmin=0,
        vmax=.05,
        xticklabels=xticks,
        yticklabels=yticks,
        mask=mask,
        cmap=cmap,
        cbar_kws={"label": "Pearson Correlation P-value"},
        ax=ax,
    )

    rotate_xaxis(ax, rotation=90)
    rotate_yaxis(ax, rotation=0)



def plot_all_bulk_pred(X, ax):
    """Plots the accuracy of the bulk prediction"""
    # Cell type percentage as input
    cell_comp_df = cell_count_perc_df(X, celltype="combined_cell_type")
    cell_comp_df = cell_comp_df.pivot(
        index=["sample_id", "binary_outcome", "patient_id", "patient_category"],
        columns="Cell Type",
        values="Cell Type Percentage",
    )
    cell_comp_df = move_index_to_column(cell_comp_df)
    cell_comp_score, _, _ = predict_mortality_all(X, cell_comp_df,
                                n_components=1, proba=False, bulk=True)
    y_cell_comp = [cell_comp_score, cell_comp_score]

    # Cell type gene expression as input
    cell_gene_df = aggregate_anndata(X)
    cell_gene_df = move_index_to_column(cell_gene_df)
    cell_gene_score, _, _ = predict_mortality_all(X, cell_gene_df, 
                                n_components=1, proba=False, bulk=True)
    y_cell_gene = [cell_gene_score, cell_gene_score]

    x = [0, 200]
    ax.axhline(y=cell_comp_score, color='red', linestyle='--')
    ax.axhline(y=cell_gene_score, color='green', linestyle='--')

