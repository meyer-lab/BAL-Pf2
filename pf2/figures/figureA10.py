"""
Figure A4: XXX
"""
import anndata
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from ..tensor import correct_conditions
from .common import getSetup
import seaborn as sns
import scanpy as sc
from pf2.data_import import combine_cell_types, add_obs
from .commonFuncs.plotPaCMAP import plot_wp_pacmap, plot_wp_per_celltype 

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((8, 8), (3, 3))

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
   

    condition_factors_df = pd.DataFrame(
        data=X.uns["Pf2_A"],
        columns=[f"Cmp. {i}" for i in np.arange(1, X.uns["Pf2_A"].shape[1] + 1)],
    )
    combine_cell_types(X)

    cmp1 = 26
    
    wprojs = X.obsm["weighted_projections"]
    thres_value= 99.5
    threshold1 = np.percentile(wprojs, thres_value, axis=0) 
    # subsetX = X[wprojs[:, cmp-1] < threshold[cmp-1], :]
    #  subsetX = X[wprojs[:, cmp-1] > threshold[cmp-1], :]
    idx = wprojs[:, cmp1-1] > threshold1[cmp1-1]
    X.obs[f"Cmp{cmp1}"] = idx
    
    cmp2 = 3
    thres_value= 99.5
    threshold = np.percentile(wprojs, thres_value, axis=0) 
    idx = wprojs[:, cmp2-1] > threshold[cmp2-1]
    X.obs[f"Cmp{cmp2}"] = idx
    
    
    
    idx = (wprojs[:, cmp1-1] > threshold[cmp1-1]) & (wprojs[:, cmp2-1] > threshold1[cmp2-1])
    X.obs["Both"] = idx
    
    
        
    print(X)
    
     
    plot_avegene_per_status_sep(
    X, 
    "KIF20A",
    ax[0])
    
    
    # plot_avegene_per_status(
    # X, 
    # "FAM111B",
    # ax[1])
    
    
    
    
    
    
    # print(smallB)
    # print(smallC)
    

    # smallA = X[X.obs["Cmp26"] == True]
    # smallB = X[X.obs["Cmp3"] == True]
    # smallC = X[(X.obs["Cmp26"] == True) & (X.obs["Cmp3"] == True)]
    
    # print(smallA)
    # print(smallB)
    # print(smallC)
    # # adata[adata.obs.cell_type == "B"]
    


    
    # sc.tl.rank_genes_groups(subsetX, "combined_cell_type", method="wilcoxon")
    # sc.pl.rank_genes_groups(subsetX, n_genes=30, save="RGG.png")
        
    
    
    return f

    
    
    
def plot_avegene_per_status(
    X: anndata.AnnData,
    gene: str,
    ax,
    condition="sample_id",
    cellType="cell_type",
    status="binary_outcome",
    status2="patient_category"
):
    """Plots average gene expression across cell types for a category of drugs"""
    genesV = X[:, gene]
    dataDF = genesV.to_df()
    dataDF = dataDF.subtract(genesV.var["means"].values)
    # dataDF["Status"] = genesV.obs[status].values
    dataDF["Condition"] = genesV.obs[condition].values
    # dataDF["Cell Type"] = genesV.obs[cellType].values
    
    
    dataDF["Cmp3"] = genesV.obs["Cmp3"].values
    dataDF["Cmp26"] = genesV.obs["Cmp26"].values
    dataDF["Both"] = genesV.obs["Both"].values
    
    dataDF.loc[(dataDF["Cmp3"] == True) & (dataDF["Cmp26"] == True), "Label"] = "Both"
    dataDF.loc[((dataDF["Cmp3"] == True) & (dataDF["Cmp26"] == False), "Label")] = "Cmp3"
    dataDF.loc[(dataDF["Cmp3"] == False) & (dataDF["Cmp26"] == True), "Label"] = "Cmp26"
    
    print(dataDF)
    dataDF = dataDF.dropna(subset="Label")
    print(dataDF)
    # # dataDF.loc[dataDF["Cmp3"] == False, "Label"] = "Neither"
    # dataDF.loc[dataDF["Cmp26"] == False, "Label"] = "Neither"
    
    
    df = pd.melt(
        dataDF, id_vars=["Label", "Condition"], value_vars=gene
    ).rename(columns={"variable": "Gene", "value": "Value"})

    df = df.groupby(["Label", "Gene", "Condition"], observed=False).mean()
    df = df.rename(columns={"Value": "Average Gene Expression"}).reset_index()
    
    print(df)
    
    sns.boxplot(
        data=df.loc[df["Gene"] == gene],
        x="Label",
        y="Average Gene Expression",
        ax=ax,
        showfliers=False,
    )
    ax.set(ylabel=f"Average {gene}")
    
  


def plot_avegene_per_status_sep(
    X: anndata.AnnData,
    gene: str,
    ax,
    condition="sample_id",
    cellType="cell_type",
    status="binary_outcome",
    status2="patient_category"
):
    """Plots average gene expression across cell types for a category of drugs"""
    genesV = X[:, gene]
    dataDF = genesV.to_df()
    dataDF = dataDF.subtract(genesV.var["means"].values)
    # dataDF["Status"] = genesV.obs[status].values
    dataDF["Condition"] = genesV.obs[condition].values
    # dataDF["Cell Type"] = genesV.obs[cellType].values
    
    
    dataDF["Cmp3"] = genesV.obs["Cmp3"].values
    dataDF["Cmp26"] = genesV.obs["Cmp26"].values
    # dataDF["Both"] = genesV.obs["Both"].values
    
    # dataDF.loc[(dataDF["Cmp3"] == True) & (dataDF["Cmp26"] == True), "Label"] = "Both"
    # dataDF.loc[((dataDF["Cmp3"] == True) & (dataDF["Cmp26"] == False), "Label")] = "Cmp3"
    # dataDF.loc[(dataDF["Cmp3"] == False) & (dataDF["Cmp26"] == True), "Label"] = "Cmp26"
    
    # print(dataDF)
    # dataDF = dataDF.dropna(subset="Label")
    # print(dataDF)
    # # dataDF.loc[dataDF["Cmp3"] == False, "Label"] = "Neither"
    # dataDF.loc[dataDF["Cmp26"] == False, "Label"] = "Neither"
    
    
    df1 = pd.melt(
        dataDF, id_vars=["Condition", "Cmp3"], value_vars=gene
    ).rename(columns={"variable": "Gene", "value": "Value"})

    df1 = df1.groupby(["Gene", "Condition", "Cmp3"], observed=False).mean()
    df1 = df1.rename(columns={"Value": "Average Gene Expression"}).reset_index()
    
    df1 = df1.loc[df1["Cmp3"] == True]

    df1["Label"] = "Cmp3"
    

    
       
    df2 = pd.melt(
        dataDF, id_vars=["Condition", "Cmp26"], value_vars=gene
    ).rename(columns={"variable": "Gene", "value": "Value"})

    df2 = df2.groupby(["Gene", "Condition", "Cmp26"], observed=False).mean()
    df2 = df2.rename(columns={"Value": "Average Gene Expression"}).reset_index()
    
    df2 = df2.loc[df2["Cmp26"] == True]
    df2["Label"] = "Cmp26"
    
    df = pd.concat([df1, df2])
    df = df.drop(columns=["Cmp3", "Cmp26"]).dropna()
    
    print(df)
    
  
    
    sns.boxplot(
        data=df,
        x="Label",
        y="Average Gene Expression",
        ax=ax,
        showfliers=False,
    )
    ax.set(ylabel=f"Average {gene}")
    
  

    
    
    # dataDF = dataDF.replace({'Status': {0: "Lived", 
    #                             1: "Dec."}})

    # dataDF["Status2"] = genesV.obs[status2].values
    # dataDF = dataDF.replace({'Status2': {"Non-Pneumonia Control": "Non-COVID", 
    #                             "Other Pneumonia": "Non-COVID",
    # #                             "Other Viral Pneumonia": "Non-COVID"}})
    # # dataDF["Status"] = dataDF["Status2"] + dataDF["Status"]

    # df = pd.melt(
    #     dataDF, id_vars=["Status", "Cell Type", "Condition"], value_vars=gene
    # ).rename(columns={"variable": "Gene", "value": "Value"})

    # df = df.groupby(["Status", "Cell Type", "Gene", "Condition"], observed=False).mean()
    # df = df.rename(columns={"Value": "Average Gene Expression"}).reset_index()
    
    # sns.boxplot(
    #     data=df.loc[df["Gene"] == gene],
    #     x="Cell Type",
    #     y="Average Gene Expression",
    #     hue="Status",
    #     ax=ax,
    #     showfliers=False,
    # )
    # ax.set(ylabel=f"Average {gene}")

    # return df

