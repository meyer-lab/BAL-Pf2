import gc

import numpy as np
import pandas as pd
from parafac2 import parafac2_nd

OPTIMAL_RANK = 40


def calc_r2x(projected, data):
    """
    Calculate the top and bottom part of R2X formula separately.

    Parameters:
        projected (np.ndarray): reconstructed tensor
        data (np.ndarray): original tensor

    Returns:
        v_top (float): variance explained
        v_bottom (float): total variance in tensor
    """
    v_top = np.linalg.norm(projected - data) ** 2.0
    v_bottom = np.linalg.norm(data) ** 2.0
    return v_top, v_bottom


def get_variance_explained(pf2, tensor):
    """
    Computes variance explained.

    Parameters:
        pf2 (tensorly.Parafac2Tensor): PF2 factorization
        tensor (np.ndarray): original dataset

    Returns:
        R2X (float): proportion of variance explained via PF2
    """
    top, bottom = 0, 0
    projected = pf2.to_tensor()
    for index, matrix in enumerate(tensor):
        projected_matrix = projected[index, :matrix.shape[0], :]
        slice_var = calc_r2x(projected_matrix, matrix)
        top += slice_var[0]
        bottom += slice_var[1]

    return 1 - top / bottom


def build_tensor(data, drop_low=100):
    """
    Builds PARAFAC2 tensor.

    Parameters:
        data (AnnData): single cell data
        drop_low (int, default:100): drops patients with fewer cells than
            drop_low

    Returns:
        tensor (np.ndarray): PARAFAC2 tensor
        labels (pd.DataFrame): maps tensor indices to patient and sample IDs
    """
    sample_ids = pd.Series((data.obs.loc[:, "sample_id"].unique()))
    tensor = []
    labels = pd.DataFrame(columns=["patient_id", "sample_id"], dtype=object)

    for index, sample_id in enumerate(sample_ids):
        matrix = data[data.obs.loc[:, "sample_id"] == sample_id, :]
        data = data[~(data.obs.loc[:, "sample_id"] == sample_id), :]
        gc.collect()

        if drop_low:
            if matrix.shape[0] >= drop_low:
                tensor.append(matrix.X)
                labels.loc[index, :] = [
                    matrix.obs.loc[:, "patient_id"].iloc[0],
                    matrix.obs.loc[:, "sample_id"].iloc[0],
                ]
        else:
            tensor.append(matrix.X)
            labels.loc[index, :] = [
                matrix.obs.loc[:, "patient_id"].iloc[0],
                matrix.obs.loc[:, "sample_id"].iloc[0],
            ]

    del data
    gc.collect()

    return tensor, labels


def run_parafac2(data, rank=OPTIMAL_RANK):
    """
    Runs PARAFAC2 on the provided data.

    Parameters:
        data (anndata.annData): single-cell dataset
        rank (int, default:DEFAULT_RANK): rank of PF2 decomposition

    Returns:
        pf2 (tensorly.Parafac2Tensor): PF2 factorization
    """
    (weights, factors, projections), r2x = parafac2_nd(
        data,
        rank=rank
    )

    data.uns['pf2'] = {}
    data.uns['pf2']['weights'] = weights
    data.uns['pf2']['factors'] = factors
    data.uns['pf2']['projections'] = projections
    data.uns['pf2']['r2x'] = r2x
    data.uns['pf2']['rank'] = rank
    
    return data, r2x
