import pandas as pd
import seaborn as sns
import anndata
from matplotlib.axes import Axes
import numpy as np


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
        palette="Set3",
    )
    ax.set(ylabel=f"Average {gene}")

    return df


def bal_combine_bo_covid(
    df, status1: str = "binary_outcome", status2: str = "patient_category"
):
    """Combines binary outcome and covid status columns"""
    df = df.replace({status1: {0: "L-", 1: "D-"}})

    df = df.replace(
        {
            status2: {
                "COVID-19": "C19",
                "Non-Pneumonia Control": "nC19",
                "Other Pneumonia": "nC19",
                "Other Viral Pneumonia": "nC19",
            }
        }
    )
    df["Status"] = df[status1] + df[status2]

    return df


def add_obs_cmp_both_label(
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
        

    if pos1 is True and pos2 is True:
        idx = (wprojs[:, cmp1 - 1] > threshold1[cmp1 - 1]) & (
            wprojs[:, cmp2 - 1] > threshold2[cmp2 - 1]
        )
    elif pos1 is False and pos2 is False:
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
        idx = (wprojs[:, cmp1 - 1] > threshold1[cmp1 - 1]) & (
            wprojs[:, cmp2 - 1] > threshold2[cmp2 - 1]) & (
                    wprojs[:, cmp3 - 1] > threshold3[cmp3 - 1]
                )
    elif pos1 is False and pos2 is False and pos3 is False:
        idx = (wprojs[:, cmp1 - 1] < threshold1[cmp1 - 1]) & (
            wprojs[:, cmp2 - 1] < threshold2[cmp2 - 1]) & (
                    wprojs[:, cmp3 - 1] < threshold3[cmp3 - 1]
                )
    elif pos1 is True and pos2 is True and pos3 is False:
        idx = (wprojs[:, cmp1 - 1] > threshold1[cmp1 - 1]) & (
            wprojs[:, cmp2 - 1] > threshold2[cmp2 - 1]) & (
                 wprojs[:, cmp3 - 1] < threshold3[cmp3 - 1]
            )

    elif pos1 is True and pos2 is False and pos3 is True:
        idx = (wprojs[:, cmp1 - 1] > threshold1[cmp1 - 1]) & (
            wprojs[:, cmp2 - 1] < threshold2[cmp2 - 1]) & (
                    wprojs[:, cmp3 - 1] > threshold3[cmp3 - 1]
                )
    elif pos1 is True and pos2 is False and pos3 is False:
        idx = (wprojs[:, cmp1 - 1] > threshold1[cmp1 - 1]) & (
            wprojs[:, cmp2 - 1] < threshold2[cmp2 - 1]) & (
                    wprojs[:, cmp3 - 1] < threshold3[cmp3 - 1]
                )
    
    elif pos1 is False and pos2 is False and pos3 is True:
        idx = (wprojs[:, cmp1 - 1] < threshold1[cmp1 - 1]) & (
            wprojs[:, cmp2 - 1] < threshold2[cmp2 - 1]) & (
                    wprojs[:, cmp3 - 1] > threshold3[cmp3 - 1]
                )
    elif pos1 is False and pos2 is True and pos3 is True:
        idx = (wprojs[:, cmp1 - 1] < threshold1[cmp1 - 1]) & (
            wprojs[:, cmp2 - 1] > threshold2[cmp2 - 1]) & (
                    wprojs[:, cmp3 - 1] > threshold3[cmp3 - 1]
                )
    elif pos1 is False and pos2 is True and pos3 is False:
        idx = (wprojs[:, cmp1 - 1] < threshold1[cmp1 - 1]) & (
            wprojs[:, cmp2 - 1] > threshold2[cmp2 - 1]) & (
                    wprojs[:, cmp3 - 1] < threshold3[cmp3 - 1]
                )

    X.obs["Both"] = idx

    return X



def add_obs_label(X: anndata.AnnData, cmp1: str, cmp2: str):
    """Creates AnnData observation column"""
    X.obs.loc[((X.obs[f"Cmp{cmp1}"] == True) & (X.obs[f"Cmp{cmp2}"] == False), "Label")] = f"Cmp{cmp1}"
    X.obs.loc[(X.obs[f"Cmp{cmp1}"] == False) & (X.obs[f"Cmp{cmp2}"] == True), "Label"] = f"Cmp{cmp2}"
    X.obs.loc[(X.obs[f"Cmp{cmp1}"] == True) & (X.obs[f"Cmp{cmp2}"] == True), "Label"] = "Both"
    X.obs.loc[(X.obs[f"Cmp{cmp1}"] == False) & (X.obs[f"Cmp{cmp2}"] == False), "Label"] = "NoLabel"
    
    return X

def add_obs_label_three(X: anndata.AnnData, cmp1: int, cmp2: int, cmp3: int):
    """Creates AnnData observation column"""
    X.obs.loc[(X.obs[f"Cmp{cmp1}"] == True) & (X.obs[f"Cmp{cmp2}"] == True)
              & (X.obs[f"Cmp{cmp3}"] == True), "Label"] = "Both"
    X.obs.loc[(X.obs[f"Cmp{cmp1}"] == False) & (X.obs[f"Cmp{cmp2}"] == False)
              & (X.obs[f"Cmp{cmp3}"] == False), "Label"] = "NoLabel"
    X.obs.loc[((X.obs[f"Cmp{cmp1}"] == True) & (X.obs[f"Cmp{cmp2}"] == False)
               & (X.obs[f"Cmp{cmp3}"] == False), "Label")] = str(f"Cmp{cmp1}")
    X.obs.loc[(X.obs[f"Cmp{cmp1}"] == False) & (X.obs[f"Cmp{cmp2}"] == True)
              & (X.obs[f"Cmp{cmp3}"] == False), "Label"] = str(f"Cmp{cmp2}")
    X.obs.loc[(X.obs[f"Cmp{cmp1}"] == False) & (X.obs[f"Cmp{cmp2}"] == False)
              & (X.obs[f"Cmp{cmp3}"] == True), "Label"] = str(f"Cmp{cmp3}")
    
    
    return X


def rotate_xaxis(ax, rotation=90):
    """Rotates text by 90 degrees for x-axis"""
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=rotation)


def rotate_yaxis(ax, rotation=90):
    """Rotates text by 90 degrees for y-axis"""
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(labels=ax.get_yticklabels(), rotation=rotation)
