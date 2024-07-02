from os.path import join
import anndata
import doubletdetection
import numpy as np
import pandas as pd
from parafac2.normalize import prepare_dataset

DATA_PATH = join("/opt", "northwest_bal")
CONVERSION_CELL_TYPES = {
    "CD8 T cells": "T Cells",
    "Monocytes1": "Monocytes",
    "Mac3 CXCL10": "Macrophages",
    "Monocytes2": "Monocytes",
    "B cells": "B Cells",
    "CD4 T cells": "T Cells",
    "CM CD8 T cells": "T Cells",
    "Tregs": "T-regulatory",
    "Plasma cells1": "B Cells",
    "Migratory DC CCR7": "Dendritic Cells",
    "Proliferating T cells": "Proliferating",
    "Monocytes3 HSPA6": "Monocytes",
    "Mac2 FABP4": "Macrophages",
    "DC2": "Dendritic Cells",
    "Mac4 SPP1": "Macrophages",
    "pDC": "Dendritic Cells",
    "Mac1 FABP4": "Macrophages",
    "Proliferating Macrophages": "Macrophages",
    "Mac6 FABP4": "Macrophages",
    "DC1 CLEC9A": "Dendritic Cells",
    "IFN resp. CD8 T cells": "T Cells",
    "NK/gdT cells": "NK Cells",
    "Mast cells": "Other",
    "Secretory cells": "Other",
    "Ciliated cells": "Other",
    "Epithelial cells": "Other",
    "Mac5 FABP4": "Macrophages",
    "Ionocytes": "Other",
}


def import_meta(drop_duplicates: bool = True) -> pd.DataFrame:
    """
    Imports meta-data.

    Parameters:
        drop_duplicates (bool, default:True): remove duplicate patients

    Returns:
        meta (pd.DataFrame): patient metadata
    """
    meta = pd.read_csv(join(DATA_PATH, "04_external.csv"), index_col=0)
    meta = meta.loc[meta.loc[:, "BAL_performed"], :]

    if drop_duplicates:
        meta = meta.loc[~meta.loc[:, "patient_id"].duplicated(), :]
        meta = meta.set_index("patient_id", drop=True)

    return meta


def import_data(small=False) -> anndata.AnnData:
    """
    Imports and preprocesses single-cell data.

    Parameters:
        small (bool, default: False): uses subset of patients
        high_variance (bool, default: True): reduce dataset to only high
            variance genes
    """
    if small:
        data = anndata.read_h5ad(
            join(DATA_PATH, "v1_01merged_cleaned_small.h5ad"),
        )
    else:
        data = anndata.read_h5ad(
            join(DATA_PATH, "v3_doublet_removed.h5ad"),
        )

    # Remove immunoglobulin genes
    data = data[:, ~data.var_names.str[:3].isin(["IGH", "IGK", "IGL"])]

    # Drop cells with high mitochondrial counts
    data = data[data.obs.pct_counts_mito < 5, :]  # type: ignore

    # Drop batch samples with few cell counts
    df = data.obs[["batch"]].reset_index(drop=True)
    df_batch = (
        df.groupby(
            ["batch"],
            observed=True
        ).size().reset_index(name="Cell Count")
    )
    batches_remove = df_batch.loc[
        df_batch["Cell Count"] <= 100
    ]["batch"].to_numpy()
    for i in batches_remove:
        data = data[data.obs["batch"] != i]

    data = prepare_dataset(data, "batch", 0.01)

    _, data.obs["condition_unique_idxs"] = np.unique(
        data.obs_vector("sample_id"),
        return_inverse=True
    )

    return data


def convert_to_patients(data: anndata.AnnData, sample: bool = False) -> pd.Series:
    """
    Converts unique IDs to patient IDs.

    Parameters:
        data (anndata.AnnData): single-cell measurements
        sample (bool): return sample ID conversion

    Returns:
        conversions (pd.Series): maps unique IDs to patient IDs.
    """
    if sample:
        conversions = data.obs.loc[:, ["sample_id", "condition_unique_idxs"]]
    else:
        conversions = data.obs.loc[:, ["patient_id", "condition_unique_idxs"]]

    conversions.set_index("condition_unique_idxs", inplace=True, drop=True)
    conversions = conversions.loc[~conversions.index.duplicated()]
    conversions.sort_index(ascending=True, inplace=True)
    conversions = conversions.squeeze()

    return conversions


def add_obs(X: anndata.annotations, new_obs: str):
    """Adds new observation based on meta and patient to individual cells"""
    meta = import_meta()
    X.obs[new_obs] = X.obs.loc[:, "patient_id"].replace(meta.loc[:, new_obs])

    return X


def obs_per_condition(X: anndata.AnnData, obs_name: str) -> pd.Series:
    """Obtain condition once only with corresponding observations"""
    all_obs = X.obs
    all_obs = all_obs.drop_duplicates(subset="condition_unique_idxs")
    all_obs = all_obs.sort_values("condition_unique_idxs")

    return all_obs[obs_name]


def combine_cell_types(X: anndata.AnnData):
    """Combined high-resolution cell types to low_resolution"""
    X.obs["combined_cell_type"] = (
        X.obs["cell_type"].map(conversion_cell_types).astype("category")
    )
    return X


def condition_factors_meta(X: anndata.AnnData):
    """Combines condition factors with meta data"""
    condition_factors = X.uns["Pf2_A"]
    meta = import_meta(drop_duplicates=False)
    meta = meta.set_index("sample_id", drop=True)
    meta = meta.loc[~meta.index.duplicated(), :]

    sample_conversions = convert_to_patients(X, sample=True)
    meta = meta.loc[meta.index.isin(sample_conversions)]
    meta = meta.reindex(sample_conversions).dropna(axis=0, how="all")
    condition_factors = condition_factors[sample_conversions.isin(meta.index), :]
    condition_factors_df = pd.DataFrame(
        index=meta.index,
        data=condition_factors,
        columns=[f"Cmp. {i}" for i in np.arange(1, condition_factors.shape[1] + 1)],
    )

    merged_df = pd.concat([condition_factors_df, meta], axis=1)

    return merged_df


conversion_cell_types = {
    "CD8 T cells": "T Cells",
    "Monocytes1": "Monocytes",
    "Mac3 CXCL10": "Macrophages",
    "Monocytes2": "Monocytes",
    "B cells": "B Cells",
    "CD4 T cells": "T Cells",
    "CM CD8 T cells": "T Cells",
    "Tregs": "T-regulatory",
    "Plasma cells1": "B Cells",
    "Migratory DC CCR7": "Dendritic Cells",
    "Proliferating T cells": "Proliferating",
    "Monocytes3 HSPA6": "Monocytes",
    "Mac2 FABP4": "Macrophages",
    "DC2": "Dendritic Cells",
    "Mac4 SPP1": "Macrophages",
    "pDC": "Dendritic Cells",
    "Mac1 FABP4": "Macrophages",
    "Proliferating Macrophages": "Macrophages",
    "Mac6 FABP4": "Macrophages",
    "DC1 CLEC9A": "Dendritic Cells",
    "IFN resp. CD8 T cells": "T Cells",
    "NK/gdT cells": "NK Cells",
    "Mast cells": "Other",
    "Secretory cells": "Other",
    "Ciliated cells": "Other",
    "Epithelial cells": "Other",
    "Mac5 FABP4": "Macrophages",
    "Ionocytes": "Other",
}

def remove_doublets(data: anndata.AnnData) -> anndata.AnnData:
    """Removes doublets."""
    data.obs.loc[:, "doublet"] = 0
    for run in data.obs.loc[:, "batch"].unique():
        sample = data.X[
                 data.obs.loc[:, "batch"] == run,
                 :
                 ]
        if sample.shape[0] < 30:
            data = data[
                   ~(data.obs.loc[:, "batch"] == run),
                   :
                   ]
            continue

        clf = doubletdetection.BoostClassifier(
            n_iters=10,
            clustering_algorithm="louvain",
            standard_scaling=True,
            pseudocount=0.1,
            n_jobs=-1,
        )
        data.obs.loc[
            data.obs.loc[:, "batch"] == run,
            "doublet"
        ] = clf.fit(sample).predict(p_thresh=1e-16, voter_thresh=0.5)

    data = data[
           ~data.obs.loc[:, "doublet"].astype(bool),
           :
           ]
    return data

