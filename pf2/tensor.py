import numpy as np
from anndata import AnnData
from pacmap import PaCMAP
from parafac2 import parafac2_nd
from tlviz.factor_tools import degeneracy_score

OPTIMAL_RANK = 40


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
):
    pf_out, r2x = parafac2_nd(data, rank=rank, random_state=random_state)

    data = store_pf2(data, pf_out)

    print(f"Degeneracy score: {degeneracy_score((pf_out[0], pf_out[1]))}")

    if do_embedding:
        pcm = PaCMAP(random_state=random_state)
        data.obsm["embedding"] = pcm.fit_transform(data.obsm["projections"])  # type: ignore

    return data, r2x
