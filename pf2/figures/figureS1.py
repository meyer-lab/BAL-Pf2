"""
Figure S1
"""

import pandas as pd
import numpy as np
import anndata
from .common import getSetup, subplotLabel
from ..tensor import correct_conditions, pf2
from ..data_import import condition_factors_meta, add_obs, combine_cell_types
from ..predict import predict_mortality_all
from ..utilities import cell_count_perc_df

def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)
    
    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    combine_cell_types(X)
    X.obs["condition_unique_idxs"] = pd.Categorical(X.obs["condition_unique_idxs"])
    ranks = np.arange(5, 70, 5)
    ranks = [2]
    r2xs = pd.Series(0, dtype=float, index=ranks)
    accuracies = pd.Series(0, dtype=float, index=ranks)
    plot_all_bulk_pred(X, ax[1])
    
    # for rank in ranks:
    #     XX, r2x = pf2(X, rank, do_embedding=False)
    #     XX.uns["Pf2_A"] = correct_conditions(XX)
    #     cond_fact_meta_df = condition_factors_meta(XX)
    #     acc, _, _ = predict_mortality_all(XX, cond_fact_meta_df, 
    #                                         n_components=1, proba=False)
    #     r2xs.loc[rank] = r2x
    #     accuracies.loc[rank] = acc
    
    # ax[0].scatter(ranks, r2xs)
    ax[0].set(xticks = ranks, ylabel = "R2X", xlabel = "Rank")
    ax[1].scatter(ranks, accuracies,)
    ax[1].set(xticks = ranks, ylabel = "Accuracy", xlabel = "Rank")
    ax[0].set(xticks=[0, 10, 20, 30, 40, 50, 60, 70])
    ax[1].set(xticks=[0, 10, 20, 30, 40, 50, 60, 70])
    
    return f



def plot_all_bulk_pred(X, ax):
    """Plots the accuracy of the bulk prediction"""
    # Cell type percentage as input
    cell_comp_df = cell_count_perc_df(X, celltype="combined_cell_type")
    cell_comp_df = cell_comp_df.pivot(
        index=["sample_id", "binary_outcome", "patient_category"],
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

    x = [0, 70]
    ax.plot(x, y_cell_comp, linestyle="--", color="r")
    ax.plot(x, y_cell_gene, linestyle="--", color="g")

    


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
                agg_values = np.ravel(agg_values)

                result_dict = {
                    "Gene": adata.var_names,
                    "Value": agg_values,
                    "Cell Type": ct,
                    "sample_id": cond,
                    "patient_category": pc,
                    "binary_outcome": bo,
                }
                results.append(pd.DataFrame(result_dict))

    df = pd.concat(results, ignore_index=True)

    pivot_df = df.pivot_table(
        index=["sample_id", "binary_outcome", "patient_category"],
        columns=["Cell Type", "Gene"],
        values=["Value"],
    )

    return pivot_df



def move_index_to_column(cell_comp_df):
    """Moves the index of a dataframe to columns"""
    bo_mapping = cell_comp_df.index.get_level_values("binary_outcome").to_numpy()
    pc_mapping = cell_comp_df.index.get_level_values("patient_category").to_numpy()
    cell_comp_df["binary_outcome"] = bo_mapping
    cell_comp_df["patient_category"] = pc_mapping

    cell_comp_df = cell_comp_df.fillna(0)
    
    return cell_comp_df