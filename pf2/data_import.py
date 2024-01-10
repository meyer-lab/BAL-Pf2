import warnings
from os.path import join
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
from sklearn.utils.sparsefuncs import inplace_column_scale, mean_variance_axis

from .tensor import pf2

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


def import_data(size="l", high_variance=True) -> anndata.AnnData:
    """
    Imports and preprocesses single-cell data.

    Parameters:
        size (str, default:'m'): size of dataset to use; must be one of 'small',
            'medium', 'large', 's', 'm', or 'l'
        high_variance (bool, default: True): reduce dataset to only high
            variance genes
    """
    if size not in ["small", "medium", "large", "s", "m", "l"]:
        size = "m"
        warnings.warn("'size' parameter not recognized; defaulting to 'medium'")

    data = anndata.read_h5ad(
        join(DATA_PATH, "v1_01merged_cleaned_qc_zm.h5ad"),
    )

    if high_variance:
        data = data[:, data.var.highly_variable]

    if size in ["small", "s"]:
        data = data[::10]
    elif size in ["medium", "m"]:
        data = data[::4]

    _, data.obs.loc[:, "condition_unique_idxs"] = np.unique(
        data.obs_vector("batch"), return_inverse=True
    )

    return data


def quality_control(
    data: anndata.AnnData, batch_correct: bool = True
) -> anndata.AnnData:
    """
    Runs single-cell dataset through quality control.

    Parameters:
        data (anndata.annData): single-cell dataset
        batch_correct (bool, default: True): correct for batches

    Returns:
        data (anndata.AnnData): quality-controlled single-cell dataset
    """
    assert isinstance(data.X, csr_matrix)

    # Drop cells with high mitochondrial counts
    data = data[data.obs.pct_counts_mito < 5, :]

    # Filter genes with few reads
    data = data[:, mean_variance_axis(data.X, axis=0)[0] > 0.002]

    # Normalize read depth
    sc.pp.normalize_total(data, exclude_highly_expressed=False, inplace=True)

    # Scale genes by sum
    inplace_column_scale(
        data.X, 1.0 / (mean_variance_axis(data.X, axis=0)[0] * data.shape[0])
    )

    # Add unique IDs
    _, data.obs.loc[:, "condition_unique_idxs"] = np.unique(
        data.obs_vector("batch"), return_inverse=True
    )

    # Log transform
    data.X.data = np.log10((1e3 * data.X.data) + 1.0)

    # Pre-compute means
    data.var["means"], _ = mean_variance_axis(data.X, axis=0)

    return data


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


def factorSave():
    data = import_data()
    factors, _ = pf2(data)
    factors.write(Path("factor_cache/factors.h5ad"))
