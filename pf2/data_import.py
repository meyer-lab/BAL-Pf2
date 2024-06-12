from os.path import join
import numpy as np
import anndata
import pandas as pd
from parafac2.normalize import prepare_dataset


DATA_PATH = join("/opt", "northwest_bal")


def import_meta() -> pd.DataFrame:
    """
    Imports meta-data.

    Returns:
         meta (pd.DataFrame): patient metadata
    """
    meta = pd.read_csv(join(DATA_PATH, "04_external.csv"), index_col=0)
    meta = meta.loc[meta.loc[:, "BAL_performed"], :]

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
            join(DATA_PATH, "v2_01merged_cleaned.h5ad"),
        )

    # Drop cells with high mitochondrial counts
    data = data[data.obs.pct_counts_mito < 5, :]  # type: ignore

    # Drop batch samples with few cell counts
    df = data.obs[["batch"]].reset_index(drop=True)
    df_batch = (
        df.groupby(["batch"], observed=True).size().reset_index(name="Cell Count")
    )
    batches_remove = df_batch.loc[df_batch["Cell Count"] <= 100]["batch"].to_numpy()
    for i in batches_remove:
        data = data[data.obs["batch"] != i]

    return prepare_dataset(data, "batch", 0.01)


def convert_to_patients(data: anndata.AnnData) -> pd.Series:
    """
    Converts unique IDs to patient IDs.

    Parameters:
        data (anndata.AnnData): single-cell measurements

    Returns:
        conversions (pd.Series): maps unique IDs to patient IDs.
    """
    conversions = data.obs.loc[:, ["patient_id", "condition_unique_idxs"]]
    conversions.set_index("condition_unique_idxs", inplace=True, drop=True)
    conversions = conversions.loc[~conversions.index.duplicated()]
    conversions.sort_index(ascending=True, inplace=True)
    conversions = conversions.squeeze()

    return conversions


def add_obs(X: anndata.annotations, new_obs: str):
    """Adds new observation based on meta and patient to inividual cells"""
    patient_id_X = np.unique(X.obs["patient_id"])
    meta = import_meta()
    reduced_meta = meta.loc[meta["patient_id"].isin(patient_id_X)][
        ["patient_id", new_obs]
    ].drop_duplicates()

    binary_outcome = np.empty(X.shape[0])
    for i, patient in enumerate(X.obs["patient_id"]):
        binary_outcome[i] = reduced_meta.loc[reduced_meta["patient_id"] == patient][
            new_obs
        ].to_numpy()

    X.obs[new_obs] = binary_outcome

    return X


def obs_per_condition(X: anndata.AnnData, obs_name: str) -> pd.DataFrame:
    """Obtain condition once only with corresponding observations"""
    all_obs = X.obs
    all_obs = all_obs.drop_duplicates(subset="condition_unique_idxs")
    all_obs = all_obs.sort_values("condition_unique_idxs")

    return all_obs[obs_name]


def combine_cell_types(X: anndata.AnnData):
    """Combined high-resolution cell types to low_resolution"""
    df = pd.DataFrame(data=X.obs["cell_type"].values)
    df = df.replace(conversion_cell_types)
    X.obs["combined_cell_type"] = np.ravel(df.values)

    return X


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
