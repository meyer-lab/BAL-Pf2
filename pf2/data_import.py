from os.path import join
from pathlib import Path

import anndata
import pandas as pd
from parafac2.normalize import prepare_dataset

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
    if small:
        data = anndata.read_h5ad(
            join(DATA_PATH, "v1_01merged_cleaned_small.h5ad"),
        )
    else:
        data = anndata.read_h5ad(
            join(DATA_PATH, "v2_01merged_cleaned.h5ad"),
        )

    # Drop cells with high mitochondrial counts
    data = data[data.obs.pct_counts_mito < 5, :] # type: ignore

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


def factorSave():
    data = import_data()
    factors, _ = pf2(data)
    factors.write(Path("factor_cache/factors.h5ad"))
