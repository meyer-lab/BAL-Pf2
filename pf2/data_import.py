import time
import warnings
from os.path import join

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
from sklearn.utils.sparsefuncs import inplace_column_scale, mean_variance_axis

DATA_PATH = join("/opt", "northwest_bal")


def import_meta():
    """
    Imports meta-data.

    Returns:
         meta (pd.DataFrame): patient metadata
    """
    meta = pd.read_csv(join(DATA_PATH, "04_external.csv"), index_col=0)
    meta = meta.loc[meta.loc[:, "BAL_performed"], :]

    return meta


def import_data(
    high_variance_genes=True,
    size="l",
) -> anndata.AnnData:
    """
    Imports and preprocesses single-cell data.

    Parameters:
        high_variance_genes (bool, default:True): use only high-variance genes
        size (str, default:'m'): size of dataset to use; must be one of 'small',
            'medium', 'large', 's', 'm', or 'l'
    """
    if size not in ["small", "medium", "large", "s", "m", "l"]:
        size = "m"
        warnings.warn("'size' parameter not recognized; defaulting to 'medium'")

    adata = anndata.read_h5ad(
        join(DATA_PATH, "v1_01merged_cleaned_qc.h5ad"),
    )
    if size in ["small", "s"]:
        adata = adata[np.arange(0, adata.shape[0], 10)]
    elif size in ["medium", "m"]:
        adata = adata[np.arange(0, adata.shape[0], 4)]

    if high_variance_genes:
        adata = adata[:, adata.var.loc[:, "highly_variable"]]

    _, adata.obs.loc[:, "condition_unique_idxs"] = np.unique(
        adata.obs_vector("batch"), return_inverse=True
    )

    return adata


def quality_control(data: anndata.AnnData, batch_correct: bool = True):
    """
    Runs single-cell dataset through quality control.

    Parameters:
        data (anndata.annData): single-cell dataset
        batch_correct (bool, default: True): correct for batches

    Returns:
        data (anndata.annData): quality-controlled single-cell dataset
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

    if batch_correct:
        data = rescale_batches(data)

    # Log transform
    data.X.data = np.log10((1e3 * data.X.data) + 1.0)

    return data


def rescale_batches(data):
    """
    Rescales batches to minimize batch effects.

    Parameters:
        data (anndata.annData): single-cell measurements

    Returns:
        data (anndata.annData): rescaled single-cell measurements
    """
    assert isinstance(data.X, csr_matrix)
    cond_labels = data.obs["condition_unique_idxs"]

    for ii in range(np.amax(cond_labels) + 1):
        xx = csr_matrix(data[cond_labels == ii].X, copy=True)

        # Scale genes by sum
        readmean = mean_variance_axis(xx, axis=0)[0]
        readsum = xx.shape[0] * readmean
        inplace_column_scale(xx, 1.0 / readsum)

        data[cond_labels == ii] = xx

    return data


def convert_to_patients(data):
    """
    Converts unique IDs to patient IDs.

    Parameters:
        data (anndata.annData): single-cell measurements

    Returns:
        conversions (pd.Series): maps unique IDs to patient IDs.
    """
    conversions = data.obs.loc[:, ["patient_id", "condition_unique_idxs"]]
    conversions.set_index("condition_unique_idxs", inplace=True, drop=True)
    conversions = conversions.loc[~conversions.index.duplicated()]
    conversions.sort_index(ascending=True, inplace=True)
    conversions = conversions.squeeze()

    return conversions
