import anndata
import numpy as np
from anndata import AnnData
from pacmap import PaCMAP
from parafac2.parafac2 import parafac2_nd, store_pf2
import scipy.cluster.hierarchy as sch
from scipy.stats import tmean
from sklearn.linear_model import LinearRegression

OPTIMAL_RANK = 40


def pf2(
    data: AnnData,
    rank: int = OPTIMAL_RANK,
    random_state=1,
    do_embedding: bool = True,
) -> tuple[anndata.AnnData, float]:
    pf_out, r2x = parafac2_nd(data, rank=rank, random_state=random_state, tol=1e-7)

    data = store_pf2(data, pf_out)

    if do_embedding:
        pcm = PaCMAP(random_state=random_state)
        data.obsm["X_pf2_PaCMAP"] = pcm.fit_transform(data.obsm["projections"])  # type: ignore

    return data, r2x


def correct_conditions(X: anndata.AnnData):
    """Correct the conditions factors by overall read depth."""
    sgIndex = X.obs["condition_unique_idxs"]
    counts = np.zeros((np.amax(sgIndex) + 1, 1))

    cond_mean = tmean(X.uns["Pf2_A"], axis=1)

    x_count = X.X.sum(axis=1)

    for ii in range(counts.size):
        counts[ii] = np.sum(x_count[X.obs["condition_unique_idxs"] == ii])

    lr = LinearRegression()
    lr.fit(counts, cond_mean.reshape(-1, 1))

    counts_correct = lr.predict(counts)

    return X.uns["Pf2_A"] / counts_correct


def reorder_table(projs: np.ndarray) -> np.ndarray:
    """Reorder a table's rows using heirarchical clustering"""
    assert projs.ndim == 2
    Z = sch.linkage(projs, method="complete", metric="cosine", optimal_ordering=True)
    return sch.leaves_list(Z)
