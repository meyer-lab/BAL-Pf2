from os.path import abspath, dirname, join

import numpy as np
import scanpy as sc
from scipy.stats import zscore
from tensorly.decomposition import parafac2


OPTIMAL_RANK = 40
REPO_PATH = dirname(abspath(__file__))


def calc_R2X(projected, data):
    """ Calculate the top and bottom part of R2X formula separately """
    v_top = np.linalg.norm(projected - data) ** 2.0
    v_bottom = np.linalg.norm(data) ** 2.0
    return v_top, v_bottom


def get_variance_explained(pf2, tensor):
    """Computes variance explained"""
    top, bottom = 0, 0
    projected = pf2.to_tensor()
    for index, matrix in enumerate(tensor):
        projected_matrix = projected[index, :matrix.shape[0], :]
        slice_var = calc_R2X(projected_matrix, matrix)
        top += slice_var[0]
        bottom += slice_var[1]

    return 1 - top / bottom


def import_data(high_variance_genes=True, sum_one=False, log_transform=True,
                normalize=True):
    """Imports and preprocesses data."""
    adata = sc.read_h5ad(
        join(
            REPO_PATH,
            'data',
            'trimmed_10f.h5ad'
        ),
        backed='r+'
    )
    if high_variance_genes:
        adata = adata[:, adata.var.loc[:, 'highly_variable']]

    data = adata.to_df()
    data.index = adata.obs.loc[:, 'library_id']

    if sum_one:
        data /= data.sum(axis=0)

    if log_transform:
        data = (data * 1E3) + 1
        data = np.log10(data)

    if normalize:
        data[:] = zscore(data, axis=0)

    return data


def build_tensor(data, drop_low=100):
    """Builds PARAFAC2 tensor."""
    patients = list(data.index.unique())
    tensor = []

    if drop_low:
        kept_patients = []

    for patient in patients:
        matrix = data.loc[
            data.index == patient,
            :
        ]
        if drop_low:
            if matrix.shape[0] >= drop_low:
                tensor.append(matrix)
                kept_patients.append(patient)
        else:
            tensor.append(matrix.values)

    if drop_low:
        return tensor, kept_patients
    else:
        return tensor, patients


def run_parafac2(data, rank=OPTIMAL_RANK):
    """Runs PARAFAC2 on the provided data."""
    pf2 = parafac2(
        data,
        rank=rank,
        init='svd',
        svd='randomized_svd',
        normalize_factors=True,
        tol=1E-6
    )
    return pf2


def main():
    data = import_data()
    tensor, patients = build_tensor(data)
    pf2 = run_parafac2(tensor)
    print(get_variance_explained(pf2, tensor))


if __name__ == '__main__':
    main()
