"""
Figure A14:
"""
from .commonFuncs.plotPaCMAP import plot_labels_pacmap
from ..data_import import combine_cell_types, add_obs
import anndata
from .common import subplotLabel, getSetup
from ..figures.commonFuncs.plotGeneral import add_obs_cmp_both_label, add_obs_label
import seaborn as sns
import matplotlib.colors as mcolors
import numpy as np
from .commonFuncs.plotFactors import bot_top_genes
from ..figures.commonFuncs.plotGeneral import bal_combine_bo_covid, rotate_xaxis
import pandas as pd
    
from ..figures.commonFuncs.plotPaCMAP import plot_gene_pacmap


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((12, 12), (4, 4))

    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    combine_cell_types(X)

    cmp1 = 15; cmp2 = 16; cmp3 = 19
    pos1 = False; pos2 = True; pos3 = False
    threshold = 0.5
    X = add_obs_cmp_both_label_three(X, cmp1, cmp2, cmp3, pos1, pos2, pos3, top_perc=threshold)
    X = add_obs_label_three(X, cmp1, cmp2, cmp3)
    
    colors = ["black", "fuchsia", "turquoise", "slateblue", "gainsboro"]
    pal = []
    for i in colors:
        pal.append(mcolors.CSS4_COLORS[i])

    plot_labels_pacmap(X, "Label", ax[0], color_key=pal)

    genes1 = bot_top_genes(X, cmp=cmp1, geneAmount=1)
    genes2 = bot_top_genes(X, cmp=cmp2, geneAmount=1)
    genes3 = bot_top_genes(X, cmp=cmp3, geneAmount=1)
    genes = np.concatenate([genes1, genes2, genes3])
    for i, gene in enumerate(genes):
        plot_avegene_cmps(X, gene, ax[i+1])
        rotate_xaxis(ax[i+1])
        
    for i, gene in enumerate(genes):
        plot_gene_pacmap(gene, X, ax[i+7])
    

    
    return f

def add_obs_cmp_both_label_three(
    X: anndata.AnnData, cmp1: int, cmp2: int, cmp3: int, pos1=True, pos2=True, pos3=True, top_perc=1
):
    """Adds if cells in top/bot percentage"""
    wprojs = X.obsm["weighted_projections"]
    pos_neg = [pos1, pos2, pos3]
    for i, cmp in enumerate([cmp1, cmp2, cmp3]):
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

        if i == 2:
            if pos_neg[i] is True:
                thres_value = 100 - top_perc
                threshold3 = np.percentile(wprojs, thres_value, axis=0)
                idx = wprojs[:, cmp - 1] > threshold3[cmp - 1]
            else:
                thres_value = top_perc
                threshold3 = np.percentile(wprojs, thres_value, axis=0)
                idx = wprojs[:, cmp - 1] < threshold3[cmp - 1]

        X.obs[f"Cmp{cmp}"] = idx

    if pos1 is True and pos2 is True and pos3 is True:
        idx = (wprojs[:, cmp1 - 1] >= threshold1[cmp1 - 1]) & (
            wprojs[:, cmp2 - 1] >= threshold2[cmp2 - 1]) & (
                    wprojs[:, cmp3 - 1] >= threshold3[cmp3 - 1]
                )
    elif pos1 is False and pos2 is False and pos3 is False:
        idx = (wprojs[:, cmp1 - 1] <= threshold1[cmp1 - 1]) & (
            wprojs[:, cmp2 - 1] <= threshold2[cmp2 - 1]) & (
                    wprojs[:, cmp3 - 1] <= threshold3[cmp3 - 1]
                )
    elif pos1 is True and pos2 is True and pos3 is False:
        idx = (wprojs[:, cmp1 - 1] >= threshold1[cmp1 - 1]) & (
            wprojs[:, cmp2 - 1] >= threshold2[cmp2 - 1]) & (
                 wprojs[:, cmp3 - 1] <= threshold3[cmp3 - 1]
            )

    elif pos1 is True and pos2 is False and pos3 is True:
        idx = (wprojs[:, cmp1 - 1] >= threshold1[cmp1 - 1]) & (
            wprojs[:, cmp2 - 1] <= threshold2[cmp2 - 1]) & (
                    wprojs[:, cmp3 - 1] >= threshold3[cmp3 - 1]
                )
    elif pos1 is True and pos2 is False and pos3 is False:
        idx = (wprojs[:, cmp1 - 1] >= threshold1[cmp1 - 1]) & (
            wprojs[:, cmp2 - 1] <= threshold2[cmp2 - 1]) & (
                    wprojs[:, cmp3 - 1] <= threshold3[cmp3 - 1]
                )

    elif pos1 is False and pos2 is False and pos3 is True:
        idx = (wprojs[:, cmp1 - 1] <= threshold1[cmp1 - 1]) & (
            wprojs[:, cmp2 - 1] <= threshold2[cmp2 - 1]) & (
                    wprojs[:, cmp3 - 1] >= threshold3[cmp3 - 1]
                )
    elif pos1 is False and pos2 is True and pos3 is True:
        idx = (wprojs[:, cmp1 - 1] <= threshold1[cmp1 - 1]) & (
            wprojs[:, cmp2 - 1] >= threshold2[cmp2 - 1]) & (
                    wprojs[:, cmp3 - 1] >= threshold3[cmp3 - 1]
                )
    elif pos1 is False and pos2 is True and pos3 is False:
        idx = (wprojs[:, cmp1 - 1] <= threshold1[cmp1 - 1]) & (
            wprojs[:, cmp2 - 1] >= threshold2[cmp2 - 1]) & (
                    wprojs[:, cmp3 - 1] <= threshold3[cmp3 - 1]
                )

    X.obs["Both"] = idx

    return X


def add_obs_label_three(X: anndata.AnnData, cmp1: int, cmp2: int, cmp3: int):
    """Creates AnnData observation column"""
    X.obs.loc[((X.obs[f"Cmp{cmp1}"] == True) & (X.obs[f"Cmp{cmp2}"] == False)
               & (X.obs[f"Cmp{cmp3}"] == False), "Label")] = str(f"Cmp{cmp1}")
    X.obs.loc[(X.obs[f"Cmp{cmp1}"] == False) & (X.obs[f"Cmp{cmp2}"] == True)
              & (X.obs[f"Cmp{cmp3}"] == False), "Label"] = str(f"Cmp{cmp2}")
    X.obs.loc[(X.obs[f"Cmp{cmp1}"] == False) & (X.obs[f"Cmp{cmp2}"] == False)
              & (X.obs[f"Cmp{cmp3}"] == True), "Label"] = str(f"Cmp{cmp3}")

    X.obs.loc[(X.obs[f"Cmp{cmp1}"] == True) & (X.obs[f"Cmp{cmp2}"] == True)
              & (X.obs[f"Cmp{cmp3}"] == True), "Label"] = str("Both")
    X.obs.loc[(X.obs[f"Cmp{cmp1}"] == False) & (X.obs[f"Cmp{cmp2}"] == False)
              & (X.obs[f"Cmp{cmp3}"] == False), "Label"] = str("NoLabel")
           
    X = X[(X.obs["Label"] == f"Cmp{cmp1}") | (X.obs["Label"] == f"Cmp{cmp2}") | 
                  (X.obs["Label"] == f"Cmp{cmp3}") | (X.obs["Label"] == "Both") |
                  (X.obs["Label"] == "NoLabel")]

    return X



def plot_avegene_cmps(
    X: anndata.AnnData,
    gene: str,
    ax,
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
        hue="Cell Type",
        ax=ax,
        # order=["Both", "CmpX", "CmpY", "NoLabel"],
        showfliers=False,
    )
    ax.set(ylabel=f"Average {gene}")

    return df
