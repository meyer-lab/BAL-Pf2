"""
Figure A4: Correlation Matrix
"""

import anndata
import numpy as np
import seaborn as sns
import pandas as pd
from .common import getSetup
from ..tensor import correct_conditions
from ..data_import import add_obs, combine_cell_types, condition_factors_meta
from ..utilities import cell_count_perc_df


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((10, 10), (2, 2))

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    factors_meta_df = condition_factors_meta(X).reset_index()
    X = add_obs(X, "binary_outcome")
    X = add_obs(X, "patient_category")
    X = X[~X.obs["patient_category"].isin(["Non-Pneumonia Control", "COVID-19"])]
    
    
    combine_cell_types(X)
    
    celltype_count_perc_df = cell_count_perc_df(X, celltype="cell_type")
    df = celltype_count_perc_df.loc[celltype_count_perc_df["Cell Type"].isin(["B cells"])]
    factors_meta_df = factors_meta_df[factors_meta_df["sample_id"].isin(df["sample_id"])]
    
    df = pd.merge(
        df,
        factors_meta_df[["sample_id"] + [f"Cmp. {i+1}" for i in range(80)]],
        on="sample_id",
        how="inner"
    )
    df = df[[f"Cmp. {i+1}" for i in range(80)]]

    pc_df = partial_correlation_matrix(df)

    print(pc_df["Cmp. 62"].sort_values(ascending=False).head(10).reset_index())
    sns.barplot(
        data=pc_df["Cmp. 62"].sort_values(ascending=False).head(10).reset_index(),
        x="index",
        y="Cmp. 62",
        ax=ax[0],

    )
    sns.barplot(
        data=pc_df["Cmp. 22"].sort_values(ascending=False).head(10).reset_index(),
        x="index",
        y="Cmp. 22",
        ax=ax[1],
    )

    return f


def partial_correlation_matrix(df: pd.DataFrame):
    """Calculates partial correlation matrix"""
    cov_df = df.cov()
    vi = np.linalg.pinv(cov_df, hermitian=True)
    vi_diag = vi.diagonal()
    D = np.diag(np.sqrt(1 / vi_diag))
    pcor = -1 * (D @ vi @ D)
    pcor[np.diag_indices_from(pcor)] = 1
    pcor_df = pd.DataFrame(pcor, columns=cov_df.columns, index=cov_df.columns)

    return pcor_df
