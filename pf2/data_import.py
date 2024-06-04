from os.path import join
from pathlib import Path

import anndata
<<<<<<< HEAD
import doubletdetection
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
from sklearn.utils.sparsefuncs import inplace_column_scale, mean_variance_axis
from tqdm import tqdm
=======
import pandas as pd
from parafac2.normalize import prepare_dataset
>>>>>>> 96d69ea4ca9293cd177aeb42e42c4d2fc670aa49

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


def import_data(small=False) -> anndata.AnnData:
    """
    Imports and preprocesses single-cell data.

    Parameters:
        small (bool, default: False): uses subset of patients
        high_variance (bool, default: True): reduce dataset to only high
            variance genes
    """
<<<<<<< HEAD
    if size not in ["small", "medium", "large", "s", "m", "l"]:
        size = "m"
        warnings.warn("'size' parameter not recognized; defaulting to 'medium'")

    data = anndata.read_h5ad(
        join(DATA_PATH, "v1_01merged_cleaned_db.h5ad"),
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
    data: anndata.AnnData,
    batch_correct: bool = True,
    remove_doublets : bool = True
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
=======
    if small:
        data = anndata.read_h5ad(
            join(DATA_PATH, "v1_01merged_cleaned_small.h5ad"),
        )
    else:
        data = anndata.read_h5ad(
            join(DATA_PATH, "v2_01merged_cleaned.h5ad"),
        )
>>>>>>> 96d69ea4ca9293cd177aeb42e42c4d2fc670aa49

    # Drop cells with high mitochondrial counts
    data = data[data.obs.pct_counts_mito < 5, :] # type: ignore

<<<<<<< HEAD
    # Filter genes with few reads
    data = data[:, mean_variance_axis(data.X, axis=0)[0] > 0.002]

    if remove_doublets:
        data.obs.loc[:, "doublet"] = 0
        for run in tqdm(data.obs.loc[:, "batch"].unique()):
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

    # Normalize read depth
    sc.pp.normalize_total(data, exclude_highly_expressed=False, inplace=True)

    # Scale genes by sum
    inplace_column_scale(
        data.X, 1.0 / (mean_variance_axis(data.X, axis=0)[0] * data.shape[0])
    )

    if batch_correct:
        data = rescale_batches(data)

    # Log transform
    data.X.data = np.log10((1e3 * data.X.data) + 1.0)

    # Pre-compute means
    data.var["means"], _ = mean_variance_axis(data.X, axis=0)

    return data


def rescale_batches(data: anndata.AnnData) -> anndata.AnnData:
    """
    Rescales batches to minimize batch effects.

    Parameters:
        data (anndata.AnnData): single-cell measurements

    Returns:
        data (anndata.AnnData): rescaled single-cell measurements
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
=======
    return prepare_dataset(data, "batch", 0.01)
>>>>>>> 96d69ea4ca9293cd177aeb42e42c4d2fc670aa49


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
