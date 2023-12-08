import numpy as np
from pacmap import PaCMAP
from anndata import AnnData
from tlviz.factor_tools import degeneracy_score
from parafac2 import parafac2_nd

OPTIMAL_RANK = 40


def store_pf2(
    X: AnnData, parafac2_output: tuple[np.ndarray, list, list]
) -> AnnData:
    """Store the Pf2 results into the anndata object."""
    sgIndex = X.obs["condition_unique_idxs"]

    X.uns["Pf2_weights"] = parafac2_output[0]
    X.uns["Pf2_A"], X.uns["Pf2_B"], X.varm["Pf2_C"] = parafac2_output[1]

    X.obsm["projections"] = np.zeros((X.shape[0], len(X.uns["Pf2_weights"])))
    for i, p in enumerate(parafac2_output[2]):
        X.obsm["projections"][sgIndex == i, :] = p  # type: ignore

    X.obsm["weighted_projections"] = X.obsm["projections"] @ X.uns["Pf2_B"]

    return X


def pf2(
    X: AnnData,
    rank: int = OPTIMAL_RANK,
    random_state=1,
    doEmbedding: bool = True,
):
    pf_out, _ = parafac2_nd(X, rank=rank, random_state=random_state)

    X = store_pf2(X, pf_out)

    print(f"Degeneracy score: {degeneracy_score((pf_out[0], pf_out[1]))}")

    if doEmbedding:
        pcm = PaCMAP(random_state=random_state)
        X.obsm["embedding"] = pcm.fit_transform(X.obsm["projections"])  # type: ignore

    return X
