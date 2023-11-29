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
