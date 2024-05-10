import anndata
import numpy as np
from anndata import AnnData
from pacmap import PaCMAP
from parafac2 import parafac2_nd
from scipy.stats import gmean
from sklearn.linear_model import LinearRegression

OPTIMAL_RANK = 43


def store_pf2(
    data: AnnData, parafac2_output: tuple[np.ndarray, list, list]
) -> AnnData:
    """Store the Pf2 results into the anndata object."""
    sg_index = data.obs["condition_unique_idxs"]

    data.uns["Pf2_weights"] = parafac2_output[0]
    data.uns["Pf2_A"], data.uns["Pf2_B"], data.varm["Pf2_C"] = parafac2_output[
        1
    ]
    data.uns["Pf2_rank"] = data.uns["Pf2_A"].shape[1]

    data.obsm["projections"] = np.zeros(
        (data.shape[0], len(data.uns["Pf2_weights"]))
    )
    for i, p in enumerate(parafac2_output[2]):
        data.obsm["projections"][sg_index == i, :] = p  # type: ignore

    data.obsm["weighted_projections"] = (
        data.obsm["projections"] @ data.uns["Pf2_B"]
    )

    return data


def pf2(
    data: AnnData,
    rank: int = OPTIMAL_RANK,
    random_state=1,
    do_embedding: bool = True,
) -> anndata.AnnData:
    pf_out, r2x = parafac2_nd(data, rank=rank, random_state=random_state, tol=1e-7)

    data = store_pf2(data, pf_out)
    # data.uns["Pf2_A"] = correct_conditions(data)

    if do_embedding:
        pcm = PaCMAP(random_state=random_state)
        data.obsm["embedding"] = pcm.fit_transform(data.obsm["projections"])  # type: ignore
        pcm = PaCMAP(random_state=random_state)
        data.uns["embedding"] = pcm.fit_transform(data.uns["Pf2_A"])  # type: ignore

    return data, r2x


def correct_conditions(X: anndata.AnnData):
    """Correct the conditions factors by overall read depth."""
    sgIndex = X.obs["condition_unique_idxs"]
    counts = np.zeros((np.amax(sgIndex) + 1, 1))

    cond_mean = gmean(X.uns["Pf2_A"], axis=1)

    x_count = X.X.sum(axis=1)

    for ii in range(counts.size):
        counts[ii] = np.sum(x_count[X.obs["condition_unique_idxs"] == ii])

    lr = LinearRegression()
    lr.fit(counts, cond_mean.reshape(-1, 1))

    counts_correct = lr.predict(counts)

    return X.uns["Pf2_A"] / counts_correct
