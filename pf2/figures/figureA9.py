"""
Figure A8:
"""

import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.axes import Axes
import anndata
from .common import subplotLabel, getSetup
from ..figures.commonFuncs.plotGeneral import bal_combine_bo_covid, rotate_xaxis
from ..data_import import add_obs, combine_cell_types
from .figureA6 import plot_cell_count, cell_count_perc_df
from .commonFuncs.plotFactors import bot_top_genes

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # ax, f = getSetup((8, 12), (7, 4))
    ax, f = getSetup((18, 18), (7, 4))
    
    subplotLabel(ax)
    
    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    combine_cell_types(X)
    
    cmp1 = 1
    cmp2 = 13
    threshold = .5
    X = add_cmp_both_label(X, cmp1, cmp2, pos1=False, pos2=False, top_perc=threshold)
    
    celltype_count_perc_df = cell_count_perc_df(X[(X.obs[f"Cmp{cmp1}"] == True) & (X.obs["Both"] == False)], celltype="combined_cell_type")
    celltype = np.unique(celltype_count_perc_df["Cell Type"])
    sns.boxplot(
        data=celltype_count_perc_df,
        x="Cell Type",
        y="Cell Count",
        hue="Status",
        order=celltype,
        showfliers=False,
        ax=ax[0],
    )
    rotate_xaxis(ax[0])
    ax[0].set(title=f"Cmp{cmp1} - Threshold: {threshold}")
    
    
    celltype_count_perc_df = cell_count_perc_df(X[(X.obs[f"Cmp{cmp2}"] == True) & (X.obs["Both"] == False)], celltype="combined_cell_type")
    celltype = np.unique(celltype_count_perc_df["Cell Type"])
    sns.boxplot(
        data=celltype_count_perc_df,
        x="Cell Type",
        y="Cell Count",
        hue="Status",
        order=celltype,
        showfliers=False,
        ax=ax[1],
    )
    rotate_xaxis(ax[1])
    ax[1].set(title=f"Cmp{cmp2} - Threshold: {threshold}")
    
    
    celltype_count_perc_df = cell_count_perc_df(X[X.obs["Both"] == True], celltype="combined_cell_type")
    celltype = np.unique(celltype_count_perc_df["Cell Type"])
    sns.boxplot(
        data=celltype_count_perc_df,
        x="Cell Type",
        y="Cell Count",
        hue="Status",
        order=celltype,
        showfliers=False,
        ax=ax[2],
    )
    rotate_xaxis(ax[2])
    ax[2].set(title=f"Both - Threshold: {threshold}")
    
    
    
    

    # plot_cell_count(X[X.obs[f"Cmp{cmp1}"] == True], ax[4])
    # ax[4].set(title=f"Cmp{cmp1}")
    # plot_cell_count(X[X.obs[f"Cmp{cmp2}"] == True], ax[5])
    # ax[5].set(title=f"Cmp{cmp2}")
    # plot_cell_count(X[X.obs["Both"] == True], ax[6])
    # ax[6].set(title="Both")

    # genes1 = bot_top_genes(X, cmp=cmp1, geneAmount=3)
    # genes2 = bot_top_genes(X, cmp=cmp2, geneAmount=3)
    
    # genes = np.concatenate([genes1, genes2])
    
    # for i, gene in enumerate(genes):
    #     plot_avegene_per_cmp(X, gene, ax[(2*i)+4], cmp1, cmp2, both_cmp = True, cell_type=True)
    #     plot_avegene_per_cmp(X, gene, ax[((2*i)+1)+4], cmp1, cmp2, both_cmp = False, cell_type=True)


    return f


def add_cmp_both_label(X: anndata.AnnData, cmp1: int, cmp2: int, pos1=True, pos2=True, top_perc=1):  
    """Adds if cells in top/bot percentage""" 
    wprojs = X.obsm["weighted_projections"]
    pos_neg = [pos1, pos2]
    
    for i, cmp in enumerate([cmp1, cmp2]):
        if i == 0:
            if pos_neg[i] is True:    
                thres_value = 100-top_perc
                threshold1 = np.percentile(wprojs, thres_value, axis=0) 
                idx = wprojs[:, cmp-1] > threshold1[cmp-1]
                
            else:
                thres_value = top_perc
                threshold1 = np.percentile(wprojs, thres_value, axis=0) 
                idx = wprojs[:, cmp-1] < threshold1[cmp-1]
                
        if i == 1:
            if pos_neg[i] is True:    
                thres_value = 100-top_perc
                threshold2 = np.percentile(wprojs, thres_value, axis=0) 
                idx = wprojs[:, cmp-1] > threshold2[cmp-1]
            else:
                thres_value = top_perc
                threshold2 = np.percentile(wprojs, thres_value, axis=0) 
                idx = wprojs[:, cmp-1] < threshold1[cmp-1]
                
        
        X.obs[f"Cmp{cmp}"] = idx

    if pos1 and pos2 is True:
        idx = (wprojs[:, cmp1-1] > threshold1[cmp1-1]) & (wprojs[:, cmp2-1] > threshold2[cmp2-1])
    elif pos1 and pos2 is False:
        idx = (wprojs[:, cmp1-1] < threshold1[cmp1-1]) & (wprojs[:, cmp2-1] < threshold2[cmp2-1])
    elif pos1 is True & pos2 is False:
        idx = (wprojs[:, cmp1-1] > threshold1[cmp1-1]) & (wprojs[:, cmp2-1] < threshold2[cmp2-1])
    elif pos1 is False & pos2 is True:
        idx = (wprojs[:, cmp1-1] < threshold1[cmp1-1]) & (wprojs[:, cmp2-1] > threshold2[cmp2-1])
        
    X.obs["Both"] = idx
    
    return X



def plot_avegene_per_cmp(
    X: anndata.AnnData,
    gene: str,
    ax: Axes,
    cmp1: int,
    cmp2: int,
    both_cmp = True,
    cell_type=False,
):
    """Plots average gene expression across cell types for a category of drugs"""
    genesV = X[:, gene]
    dataDF = genesV.to_df()
    condition="sample_id"
    status1="binary_outcome"
    status2="patient_category"
    cellType="combined_cell_type"
    
    dataDF = dataDF.subtract(genesV.var["means"].values)
    dataDF[status1] = genesV.obs[status1].values
    dataDF[status2] = genesV.obs[status2].values
    dataDF["Condition"] = genesV.obs[condition].values
    dataDF["Cell Type"] = genesV.obs[cellType].values
    
    dataDF[f"Cmp{cmp1}"] = genesV.obs[f"Cmp{cmp1}"].values
    dataDF[f"Cmp{cmp2}"] = genesV.obs[f"Cmp{cmp2}"].values
    if both_cmp is True:
        dataDF["Both"] = genesV.obs["Both"].values

    dataDF.loc[((dataDF[f"Cmp{cmp1}"] == True) & (dataDF[f"Cmp{cmp2}"] == False), "Label")] = f"Cmp{cmp1}"
    dataDF.loc[(dataDF[f"Cmp{cmp1}"] == False) & (dataDF[f"Cmp{cmp2}"] == True), "Label"] = f"Cmp{cmp2}"
    if both_cmp is True:
        dataDF.loc[(dataDF[f"Cmp{cmp1}"] == True) & (dataDF[f"Cmp{cmp2}"] == True), "Label"] = "Both"
    
    dataDF = dataDF.dropna(subset="Label")
    dataDF = bal_combine_bo_covid(dataDF, status1, status2)
    
    
    if cell_type is True:
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

    if cell_type is False:
        df = pd.melt(
            dataDF, id_vars=["Label", "Condition"], value_vars=gene
        ).rename(columns={"variable": "Gene", "value": "Value"})

        df = df.groupby(["Label", "Gene", "Condition"], observed=False).mean()
        df = df.rename(columns={"Value": "Average Gene Expression"}).reset_index()

        sns.boxplot(
            data=df.loc[df["Gene"] == gene],
            x="Label",
            y="Average Gene Expression",
            hue="Label",
            ax=ax,
            showfliers=False,
        )
        ax.set(ylabel=f"Average {gene}")

    return df