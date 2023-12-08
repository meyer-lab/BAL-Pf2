import time
import warnings
from os.path import join
from scipy.sparse import csr_matrix
from sklearn.utils.sparsefuncs import inplace_column_scale, mean_variance_axis

import anndata
import numpy as np
import pandas as pd
import scanpy as sc

DATA_PATH = join('/opt', 'northwest_bal')


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
):
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
        join(DATA_PATH, 'v1_01merged_cleaned_qc.h5ad'),
    )
    if size in ["small", "s"]:
        adata = adata[np.arange(0, adata.shape[0], 10)]
    elif size in ["medium", "m"]:
        adata = adata[np.arange(0, adata.shape[0], 4)]

    if high_variance_genes:
        adata = adata[:, adata.var.loc[:, "highly_variable"]]

    _, adata.obs.loc[:, "condition_unique_idxs"] = np.unique(
        adata.obs_vector('batch'),
        return_inverse=True
    )

    return adata


def quality_control(data, filter_low=True, log_norm=True, batch_correct=True):
    """
    Runs single-cell dataset through quality control.

    Parameters:
        data (anndata.annData): single-cell dataset
        filter_low (bool, default: True): filter cells/genes with low counts
        log_norm (bool, default: True): log-normalize genes
        batch_correct (bool, default: True): correct for batches

    Returns:
        data (anndata.annData): quality-controlled single-cell dataset
    """
    assert isinstance(data.X, csr_matrix)

    # Drop cells with high mitochondrial counts
    data = data[data.obs.pct_counts_mito < 5, :]

    if filter_low:
        # Drop cells & genes with low counts
        start = time.time()
        sc.pp.filter_cells(data, min_genes=data.n_vars // 1E2)
        sc.pp.filter_genes(data, min_cells=data.n_obs // 1E3)
        print(f'Filtering completed in {round(time.time() - start, 2)} seconds')
        
    if log_norm:
        # Log normalize
        start = time.time()
        sc.pp.normalize_total(data, target_sum=1E4)
        print(f'Log-normalization completed in {round(time.time() - start, 2)} '
              'seconds')

    if batch_correct:
        # Batch correction via ComBat
        start = time.time()
        data = rescale_batches(data)
        print(f'ComBat completed in {round(time.time() - start, 2)} seconds')
        
    sc.pp.log1p(data)

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
    conversions = data.obs.loc[:, ['patient_id', 'condition_unique_idxs']]
    conversions.set_index('condition_unique_idxs', inplace=True, drop=True)
    conversions = conversions.loc[~conversions.index.duplicated()]
    conversions.sort_index(ascending=True, inplace=True)
    conversions = conversions.squeeze()

    return conversions
