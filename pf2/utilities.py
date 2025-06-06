import anndata
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from typing import List, Dict
import scipy.stats as stats
import statsmodels.api as sm

def bal_combine_bo_covid(
    df, status1: str = "binary_outcome", status2: str = "patient_category"
):
    """Combines binary outcome and covid status columns"""
    df["binary_outcome_str"] = df[status1].map({0: "L-", 1: "D-"})

    
    df["combined_patient_category"] = df[status2].map(
            {"COVID-19": "C19",
                "Non-Pneumonia Control": "Ctrl",
                "Other Pneumonia": "nC19",
                "Other Viral Pneumonia": "nC19",
            }
    )
    df["Uncombined"] = df["binary_outcome_str"] + df[status2]
    df["Status"] = df["binary_outcome_str"] + df["combined_patient_category"]

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



def cell_count_perc_df(X, celltype="Cell Type", include_control=True):
    """Returns DF with cell counts and percentages for experiment"""
    grouping = [celltype, "sample_id", "binary_outcome", "patient_category", "patient_id"]
    df = X.obs[grouping].reset_index(drop=True)
    if include_control is False:
         df = df[df["patient_category"] != "Non-Pneumonia Control"]
    df = bal_combine_bo_covid(df)

    bo_mapping = X.obs.groupby("sample_id", observed=False)["binary_outcome"].first()
    pc_mapping = X.obs.groupby("sample_id", observed=False)["patient_category"].first()
    pid_mapping = X.obs.groupby(
        "sample_id",
        observed=False
    )["patient_id"].first()
 
    dfCond = (
        df.groupby(["sample_id"], observed=True).size().reset_index(name="Cell Count")
    )
    dfCellType = (
        df.groupby([celltype, "sample_id", "Status"], observed=True)
        .size()
        .reset_index(name="Cell Count")
    )
    dfCellType["Cell Count"] = dfCellType["Cell Count"].astype("float")

    dfCellType["Cell Type Percentage"] = 0.0
    for cond in np.unique(df["sample_id"]):
        dfCellType.loc[dfCellType["sample_id"] == cond, "Cell Type Percentage"] = (
            100
            * dfCellType.loc[dfCellType["sample_id"] == cond, "Cell Count"].to_numpy()
            / dfCond.loc[dfCond["sample_id"] == cond]["Cell Count"].to_numpy()
        )

    dfCellType["binary_outcome"] = dfCellType["sample_id"].map(bo_mapping)
    dfCellType["patient_category"] = dfCellType["sample_id"].map(pc_mapping)
    dfCellType["patient_id"] = dfCellType["sample_id"].map(pid_mapping)
    dfCellType.rename(columns={celltype: "Cell Type"}, inplace=True)

    return dfCellType



def aggregate_anndata(adata):
    """Aggregate AnnData object by cell type and condition."""
    cell_types = adata.obs["combined_cell_type"].unique()
    conditions = adata.obs["sample_id"].unique()
    results = []

    for ct in cell_types:
        for cond in conditions:
            mask = (adata.obs["combined_cell_type"] == ct) & (adata.obs["sample_id"] == cond)
            group_data = adata[mask]
            if group_data.shape[0] > 0:
                agg_values = np.mean(group_data.X, axis=0)
                bo = group_data.obs["binary_outcome"].unique()[0]
                pc = group_data.obs["patient_category"].unique()[0]
                pid = group_data.obs["patient_id"].unique()[0]
                agg_values = np.ravel(agg_values)

                result_dict = {
                    "Gene": adata.var_names,
                    "Value": agg_values,
                    "Cell Type": ct,
                    "sample_id": cond,
                    "patient_category": pc,
                    "binary_outcome": bo,
                    "patient_id": pid
                }
                results.append(pd.DataFrame(result_dict))

    df = pd.concat(results, ignore_index=True)

    pivot_df = df.pivot_table(
        index=["sample_id", "binary_outcome", "patient_category", "patient_id"],
        columns=["Cell Type", "Gene"],
        values=["Value"],
    )

    return pivot_df


def move_index_to_column(cell_comp_df):
    """Moves the index of a dataframe to columns"""
    bo_mapping = cell_comp_df.index.get_level_values("binary_outcome").to_numpy()
    pc_mapping = cell_comp_df.index.get_level_values("patient_category").to_numpy()
    pid_mapping = cell_comp_df.index.get_level_values("patient_id").to_numpy()
    cell_comp_df["binary_outcome"] = bo_mapping
    cell_comp_df["patient_category"] = pc_mapping
    cell_comp_df["patient_id"] = pid_mapping

    cell_comp_df = cell_comp_df.fillna(0)
    
    return cell_comp_df


def get_significance(p_val: float) -> str:
    """Convert p-value to significance category.
    """
    if p_val >= 0.05:
        return "NS"
    elif p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    return "NS"


def perform_statistical_tests(
    df: pd.DataFrame,
    celltype: str,
    comparisons: List[Dict[str, str]],
    ref_col: str = "Status",
    value_col: str = "Cell Type Percentage"
) -> pd.DataFrame:
    """Perform statistical tests for cell type comparisons.
    """
    pvalue_results = []
    
    for comparison in comparisons:
        filtered_df = df[df[ref_col].isin(comparison["groups"])]
        celltype_order = np.unique(filtered_df["Cell Type"])
        
        for cell_type in celltype_order:
            cell_type_data = filtered_df[filtered_df["Cell Type"] == cell_type]
            if len(cell_type_data) > 0:
                pval_df = wls_stats_comparison(
                    cell_type_data, 
                    value_col, 
                    ref_col, 
                    comparison["ref"]
                )
                
                p_val = pval_df["p Value"].iloc[0]
                significance = get_significance(p_val)
                
                pvalue_results.append({
                    'Cell Type': cell_type,
                    'Classification': celltype,
                    'P_Value': p_val,
                    'Significance': significance,
                    'Comparison': comparison["name"],
                    'Ref_Group': comparison["ref"]
                })
    
    return pd.DataFrame(pvalue_results)


def wls_stats_comparison(df, column_comparison_name, category_name, status_name):
    """Calculates whether cells are statistically significantly different using WLS"""
    pval_df = pd.DataFrame()
    df = df.copy()
    df.loc[:, "Y"] = 1 
    df.loc[df[category_name] == status_name, "Y"] = 0

    Y = df[column_comparison_name].to_numpy()
    X = df["Y"].to_numpy()
    weights = np.power(df["Cell Count"].values, 1)

    mod_wls = sm.WLS(Y, sm.tools.tools.add_constant(X), weights=weights)
    res_wls = mod_wls.fit()

    pval_df = pd.DataFrame({
        "Cell Type": ["Other"],
        "p Value": [res_wls.pvalues[1]]
    })

    return pval_df