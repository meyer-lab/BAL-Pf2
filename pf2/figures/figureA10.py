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
from pf2.figures.commonFuncs.plotGeneral import rotate_xaxis, bal_combine_bo_covid
from ..data_import import add_obs, condition_factors_meta, combine_cell_types


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((8, 8), (3, 3))

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
   

    # condition_factors_df = pd.DataFrame(
    #     data=X.uns["Pf2_A"],
    #     columns=[f"Cmp. {i}" for i in np.arange(1, X.uns["Pf2_A"].shape[1] + 1)],
    # )
    combine_cell_types(X)

    cmp1 = 28
    
    # Need to change threshold and inequalites for eveyr component 
    
    wprojs = X.obsm["weighted_projections"]
    thres_value= 99
    threshold1 = np.percentile(wprojs, thres_value, axis=0) 
    # subsetX = X[wprojs[:, cmp-1] < threshold[cmp-1], :]
    #  subsetX = X[wprojs[:, cmp-1] > threshold[cmp-1], :]
    # idx = wprojs[:, cmp1-1] > threshold1[cmp1-1]
    idx = wprojs[:, cmp1-1] > threshold1[cmp1-1]
    X.obs[f"Cmp{cmp1}"] = idx
    
    cmp2 = 45
    thres_value=1
    threshold2 = np.percentile(wprojs, thres_value, axis=0) 
    idx = wprojs[:, cmp2-1] < threshold2[cmp2-1]
    # idx = wprojs[:, cmp1-1] < threshold1[cmp2-1]
    X.obs[f"Cmp{cmp2}"] = idx
    
    
    # Need to change threshold and inequalites for eveyr component 
    idx = (wprojs[:, cmp1-1] > threshold1[cmp1-1]) & (wprojs[:, cmp2-1] < threshold2[cmp2-1])
    X.obs["Both"] = idx
    

        # print(X)
    
     
    df = plot_avegene_per_status_both(
    X, 
    "AGER",
    ax[0],
     cmp1,
    cmp2)
    df = plot_avegene_per_status_sep(
    X, 
    "AGER",
    ax[1],
     cmp1,
    cmp2)
    df = plot_avegene_per_status_both(
    X, 
    "SFN",
    ax[2],
    cmp1,
    cmp2)
    df = plot_avegene_per_status_sep(
    X, 
    "SFN",
    ax[3],
      cmp1,
    cmp2)
    df = plot_avegene_per_status_both(
    X, 
    "TM4SF1",
    ax[4],
      cmp1,
    cmp2)
    df = plot_avegene_per_status_sep(
    X, 
    "TM4SF1",
    ax[5],
    cmp1,
    cmp2)
    
    
    print(X[X.obs["Both"] == True])
    print(X[X.obs[f"Cmp{cmp1}"] == True])
    
    
    plot_cell_count(X[X.obs[f"Cmp{cmp1}"] == True], ax[6])
    ax[6].set(title=f"Cmp{cmp1}")
    plot_cell_count(X[X.obs[f"Cmp{cmp2}"] == True], ax[7])
    ax[7].set(title=f"Cmp{cmp2}")
    plot_cell_count(X[X.obs["Both"] == True], ax[8])
    ax[8].set(title="Both")
        

    
    
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

    
    
    
def plot_avegene_per_status_both(
    X: anndata.AnnData,
    gene: str,
    ax,
    cmp1,
    cmp2,
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
    
    
    dataDF[f"Cmp{cmp1}"] = genesV.obs[f"Cmp{cmp1}"].values
    dataDF[f"Cmp{cmp2}"] = genesV.obs[f"Cmp{cmp2}"].values
    dataDF["Both"] = genesV.obs["Both"].values
    
    dataDF.loc[(dataDF[f"Cmp{cmp1}"] == True) & (dataDF[f"Cmp{cmp2}"] == True), "Label"] = "Both"
    dataDF.loc[((dataDF[f"Cmp{cmp1}"] == True) & (dataDF[f"Cmp{cmp2}"] == False), "Label")] = f"Cmp{cmp1}"
    dataDF.loc[(dataDF[f"Cmp{cmp1}"] == False) & (dataDF[f"Cmp{cmp2}"] == True), "Label"] = f"Cmp{cmp2}"

    dataDF = dataDF.dropna(subset="Label")
    
    df = pd.melt(
        dataDF, id_vars=["Label", "Condition"], value_vars=gene
    ).rename(columns={"variable": "Gene", "value": "Value"})

    df = df.groupby(["Label", "Gene", "Condition"], observed=False).mean()
    df = df.rename(columns={"Value": "Average Gene Expression"}).reset_index()
    
    
    sns.boxplot(
        data=df.loc[df["Gene"] == gene],
        x="Label",
        y="Average Gene Expression",
        ax=ax,
        showfliers=False,
    )
    ax.set(ylabel=f"Average {gene}")
    
    return df
    
  


def plot_avegene_per_status_sep(
    X: anndata.AnnData,
    gene: str,
    ax,
    cmp1,
    cmp2,
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
    
    
    dataDF[f"Cmp{cmp1}"] = genesV.obs[f"Cmp{cmp1}"].values
    dataDF[f"Cmp{cmp2}"] = genesV.obs[f"Cmp{cmp2}"].values


    df1 = pd.melt(
        dataDF, id_vars=["Condition", f"Cmp{cmp1}"], value_vars=gene
    ).rename(columns={"variable": "Gene", "value": "Value"})

    df1 = df1.groupby(["Gene", "Condition", f"Cmp{cmp1}"], observed=False).mean()
    df1 = df1.rename(columns={"Value": "Average Gene Expression"}).reset_index()
    
    df1 = df1.loc[df1[f"Cmp{cmp1}"] == True]
    df1["Label"] = f"Cmp{cmp1}"
      

    
    df2 = pd.melt(
        dataDF, id_vars=["Condition", f"Cmp{cmp2}"], value_vars=gene
    ).rename(columns={"variable": "Gene", "value": "Value"})

    df2 = df2.groupby(["Gene", "Condition", f"Cmp{cmp2}"], observed=False).mean()
    df2 = df2.rename(columns={"Value": "Average Gene Expression"}).reset_index()
    
    df2 = df2.loc[df2[f"Cmp{cmp2}"] == True]
    df2["Label"] = f"Cmp{cmp2}"
    
    df = pd.concat([df1, df2])
    df = df.drop(columns=[f"Cmp{cmp1}", f"Cmp{cmp2}"]).dropna()
    
    sns.boxplot(
        data=df,
        x="Label",
        y="Average Gene Expression",
        ax=ax,
        showfliers=False,
    )
    ax.set(ylabel=f"Average {gene}")
    
    
    return df
    
  
  
          
def plot_cell_count(X: anndata.AnnData, ax, cond: str = "sample_id",
                    status1: str = "binary_outcome", status2: str = "patient_category"):
    """Plots overall cell count."""
    df = X.obs[[cond, status1, status2]].reset_index(drop=True)
    
    df = bal_combine_bo_covid(df)
    dfCond = df.groupby([cond, "Status"], observed=True).size().reset_index(name="Cell Count")

    sns.barplot(data=dfCond, x="Status", y="Cell Count", hue="Status", ax=ax)
    rotate_xaxis(ax)

    
    
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

