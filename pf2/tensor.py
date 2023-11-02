import gc

from data_import import import_data
import numpy as np
import pandas as pd
from tensorly.decomposition import parafac2

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
    sample_ids = pd.Series((data.obs.loc[:, 'sample_id'].unique()))
    tensor = []
    labels = pd.DataFrame(
        columns=['patient_id', 'sample_id'],
        dtype=object
    )

    for index, sample_id in enumerate(sample_ids):
        matrix = data[
            data.obs.loc[:, 'sample_id'] == sample_id,
            :
        ]
        data = data[
            ~(data.obs.loc[:, 'sample_id'] == sample_id),
            :
        ]
        gc.collect()

        if drop_low:
            if matrix.shape[0] >= drop_low:
                tensor.append(matrix.X)
                labels.loc[index, :] = [
                    matrix.obs.loc[:, 'patient_id'].iloc[0],
                    matrix.obs.loc[:, 'sample_id'].iloc[0]
                ]
        else:
            tensor.append(matrix.X)
            labels.loc[index, :] = [
                matrix.obs.loc[:, 'patient_id'].iloc[0],
                matrix.obs.loc[:, 'sample_id'].iloc[0]
            ]

    del data
    gc.collect()

    return tensor, labels


def run_parafac2(tensor, rank=OPTIMAL_RANK):
    """
    Runs PARAFAC2 on the provided data.

    Parameters:
        tensor (np.ndarray): PARAFAC2 tensor
        rank (int, default:DEFAULT_RANK): rank of PF2 decomposition

    Returns:
        pf2 (tensorly.Parafac2Tensor): PF2 factorization
    """
    pf2 = parafac2(
        tensor,
        rank=rank,
        init='svd',
        svd='randomized_svd',
        normalize_factors=True,
        tol=1E-6
    )
    return pf2


def pf2_low_memory(data_params=None, drop_low=100, rank=OPTIMAL_RANK):
    """
    Combines data import, tensor building, PF2 for reduced memory usage.

    Parameters:
        data_params (dict, default:None): parameters to pass to import_data
        drop_low (int, default:100): drops patients with fewer cells than
            drop_low
        rank (int, default:DEFAULT_RANK): rank of PF2 decomposition

    Returns:
        pf2 (tensorly.Parafac2Tensor): PF2 factorization
        labels (pd.DataFrame): maps tensor indices to patient and sample IDs
    """
    if data_params is None:
        data_params = {}

    data = import_data(**data_params)
    sample_ids = pd.Series((data.obs.loc[:, 'sample_id'].unique()))
    tensor = []
    labels = pd.DataFrame(
        columns=['patient_id', 'sample_id'],
        dtype=object
    )

    for index, sample_id in enumerate(sample_ids):
        matrix = data[
                 data.obs.loc[:, 'sample_id'] == sample_id,
                 :
                 ]
        data = data[
               ~(data.obs.loc[:, 'sample_id'] == sample_id),
               :
               ]
        gc.collect()

        if drop_low:
            if matrix.shape[0] >= drop_low:
                tensor.append(matrix.X)
                labels.loc[index, :] = [
                    matrix.obs.loc[:, 'patient_id'].iloc[0],
                    matrix.obs.loc[:, 'sample_id'].iloc[0]
                ]
        else:
            tensor.append(matrix.X)
            labels.loc[index, :] = [
                matrix.obs.loc[:, 'patient_id'].iloc[0],
                matrix.obs.loc[:, 'sample_id'].iloc[0]
            ]

    del data
    gc.collect()

    pf2 = parafac2(
        tensor,
        rank=rank,
        init='svd',
        svd='randomized_svd',
        normalize_factors=True,
        tol=1E-6
    )

    return pf2, labels
