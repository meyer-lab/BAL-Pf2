import anndata
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch


def bal_combine_bo_covid(
    df, status1: str = "binary_outcome", status2: str = "patient_category"
):
    """Combines binary outcome and covid status columns"""
    df = df.replace({status1: {0: "L-", 1: "D-"}})

    df = df.replace(
        {
            status2: {
                "COVID-19": "C19",
                "Non-Pneumonia Control": "Ctrl",
                "Other Pneumonia": "nC19",
                "Other Viral Pneumonia": "nC19",
            }
        }
    )
    df["Status"] = df[status1] + df[status2]

    return df


def reorder_table(projs: np.ndarray) -> np.ndarray:
    """Reorder a table's rows using heirarchical clustering"""
    assert projs.ndim == 2
    Z = sch.linkage(projs, method="complete", metric="cosine", optimal_ordering=True)
    return sch.leaves_list(Z)


def bot_top_genes(X, cmp, geneAmount=5):
    """Saves most pos/negatively genes"""
    df = pd.DataFrame(
        data=X.varm["Pf2_C"][:, cmp - 1], index=X.var_names, columns=["Component"]
    )
    df = df.reset_index(names="Gene")
    df = df.sort_values(by="Component")

    top = df.iloc[-geneAmount:, 0].values
    bot = df.iloc[:geneAmount, 0].values
    all_genes = np.concatenate([bot, top])

    return all_genes


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
                idx = wprojs[:, cmp - 1] < threshold2[cmp - 1]

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


def add_obs_cmp_unique_two(X: anndata.AnnData, cmp1: str, cmp2: str):
    """Creates AnnData observation column"""
    X.obs.loc[((X.obs[f"Cmp{cmp1}"] == True) & (X.obs[f"Cmp{cmp2}"] == False), "Label")] = f"Cmp{cmp1}"
    X.obs.loc[(X.obs[f"Cmp{cmp1}"] == False) & (X.obs[f"Cmp{cmp2}"] == True), "Label"] = f"Cmp{cmp2}"
    X.obs.loc[(X.obs[f"Cmp{cmp1}"] == True) & (X.obs[f"Cmp{cmp2}"] == True), "Label"] = "Both"
    X.obs.loc[(X.obs[f"Cmp{cmp1}"] == False) & (X.obs[f"Cmp{cmp2}"] == False), "Label"] = "NoLabel"
    
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
                idx = wprojs[:, cmp - 1] < threshold2[cmp - 1]

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


def add_obs_cmp_unique_three(X: anndata.AnnData, cmp1: int, cmp2: int, cmp3: int):
    """Creates AnnData observation column"""
    X.obs.loc[((X.obs[f"Cmp{cmp1}"] == True) & (X.obs[f"Cmp{cmp2}"] == False)
               & (X.obs[f"Cmp{cmp3}"] == False), "Label")] = f"Cmp{cmp1}"
    X.obs.loc[(X.obs[f"Cmp{cmp1}"] == False) & (X.obs[f"Cmp{cmp2}"] == True)
              & (X.obs[f"Cmp{cmp3}"] == False), "Label"] = f"Cmp{cmp2}"
    X.obs.loc[(X.obs[f"Cmp{cmp1}"] == False) & (X.obs[f"Cmp{cmp2}"] == False)
              & (X.obs[f"Cmp{cmp3}"] == True), "Label"] = f"Cmp{cmp3}"

    X.obs.loc[(X.obs[f"Cmp{cmp1}"] == True) & (X.obs[f"Cmp{cmp2}"] == True)
              & (X.obs[f"Cmp{cmp3}"] == True), "Label"] = "Both"
    X.obs.loc[(X.obs[f"Cmp{cmp1}"] == False) & (X.obs[f"Cmp{cmp2}"] == False)
              & (X.obs[f"Cmp{cmp3}"] == False), "Label"] = "NoLabel"
           
    X = X[(X.obs["Label"] == f"Cmp{cmp1}") | (X.obs["Label"] == f"Cmp{cmp2}") | 
                  (X.obs["Label"] == f"Cmp{cmp3}") | (X.obs["Label"] == "Both") |
                  (X.obs["Label"] == "NoLabel")]

    return X
