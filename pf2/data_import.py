import time
import warnings
from os.path import abspath, dirname, join

import numpy as np
import pandas as pd
import scanpy as sc

DATA_PATH = join('/opt', 'northwest_bal')
REPO_PATH = dirname(abspath(__file__))


def import_meta():
    """
    Imports meta-data.

    Returns:
         meta (pd.DataFrame): patient metadata
    """
    meta = pd.read_csv(join(REPO_PATH, "data", "04_external.csv"), index_col=0)
    meta = meta.loc[meta.loc[:, "BAL_performed"], :]

    return meta


def import_data(
    high_variance_genes=True,
    sum_one=False,
    log_transform=False,
    normalize=True,
    size="m",
):
    """
    Imports and preprocesses single-cell data.

    Parameters:
        high_variance_genes (bool, default:True): use only high-variance genes
        sum_one (bool, default:False): sums each gene to 1 across cells
        log_transform (bool, default:True): log transform data
        normalize (bool, default:True): zero mean, unit variance genes
        size (str, default:'m'): size of dataset to use; must be one of 'small',
            'medium', 'large', 's', 'm', or 'l'
    """
    if size not in ["small", "medium", "large", "s", "m", "l"]:
        size = "m"
        warnings.warn("'size' parameter not recognized; defaulting to 'medium'")

    adata = sc.read_h5ad(
        join(DATA_PATH, 'v4_11integrated_cleaned.h5ad'),
    )
    if size in ["small", "s"]:
        adata = adata[np.arange(0, adata.shape[0], 10)]
    elif size in ["medium", "m"]:
        adata = adata[np.arange(0, adata.shape[0], 4)]

    if high_variance_genes:
        adata = adata[:, adata.var.loc[:, "highly_variable"]]

    if sum_one:
        adata.X /= adata.X.sum(axis=0)

    if log_transform:
        sc.pp.log1p(adata)

    if normalize:
        sc.pp.scale(adata)

    return adata


def quality_control(data, filter_low=True, mito=True, log_norm=True, 
    scale=True, batch_correct=True):
    """
    Runs single-cell dataset through quality control.

    Parameters:
        data (anndata.annData): single-cell dataset
        filter_low (bool, default: True): filter cells/genes with low counts
        mito (bool, default: True): remove cells with high mitochondrial
            transcripts
        log_norm (bool, default: True): log-normalize genes
        scale (bool, default: True): zero mean, unit variance genes
        batch_correct (bool, default: True): correct for batches

    Returns:
        data (pd.DataFrame): quality-controlled single-cell dataset
    """
    if filter_low:
        # Drop cells & genes with low counts
        start = time.time()
        sc.pp.filter_cells(data, min_genes=data.n_vars // 1E2)
        sc.pp.filter_genes(data, min_cells=data.n_obs // 1E3)
        print(f'Filtering completed in {round(time.time() - start, 2)} seconds')

    if mito:
        # Drop cells with high mitochondrial counts
        start = time.time()
        data = data[data.obs.pct_counts_mito < 5, :]
        print(f'Mitochondrial filtering completed in {round(time.time() - start, 2)} seconds')

    if log_norm:
        # Log normalize
        start = time.time()
        sc.pp.normalize_total(data, target_sum=1E4)
        sc.pp.log1p(data)
        print(f'Log-normalization completed in {round(time.time() - start, 2)} seconds')

    if scale:
        # Zero mean, unit variance
        start = time.time()
        sc.pp.scale(data, max_value=10, zero_center=False)
        print(f'Z-score completed in {round(time.time() - start, 2)} seconds')

    if batch_correct:
        # Batch correction via ComBat
        start = time.time()
        sc.pp.combat(data)
        print(f'ComBat completed in {round(time.time() - start, 2)} seconds')

    return data
