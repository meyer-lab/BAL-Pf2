"""Figure J7b: Protective TCR Subsets"""
from os.path import join

import numpy as np
from anndata import read_h5ad

from pf2.data_import import import_meta
from pf2.figures.common import getSetup

DATA_PATH = join("/opt", "northwest_bal")

COMPONENTS = np.array([22, 62])


def makeFigure():
    data = read_h5ad("/opt/northwest_bal/full_fitted.h5ad", backed="r")

    meta = import_meta(sample_index=True)
    meta = meta.loc[data.obs.loc[:, "sample_id"].unique(), :]
    meta.loc[:, "covid_status"] = meta.loc[
        :,
        "covid_status"
    ].fillna(False)
    meta = meta.loc[
        meta.loc[:, "covid_status"],
        :
    ]
    meta = meta.loc[~meta.loc[:, "episode_category"].isna(), :]

    cap = meta.loc[meta.loc[:, "episode_category"] == "CAP"]
    n_cap = meta.loc[meta.loc[:, "episode_category"] != "CAP"]

    axs, fig = getSetup(
        (4, 4),
        (1, 1)
    )
    ax = axs[0]

    ax.errorbar(
        0,
        cap.loc[:, "binary_outcome"].mean(),
        yerr=np.clip(
            cap.loc[:, "binary_outcome"].std() * 1.96 / np.sqrt(cap.shape[0]),
            0,
            1
        ),
        marker=".",
        capsize=2
    )
    ax.errorbar(
        1,
        n_cap.loc[:, "binary_outcome"].mean(),
        yerr=n_cap.loc[:, "binary_outcome"].std() * 1.96 / np.sqrt(
            n_cap.shape[0]
        ),
        marker=".",
        capsize=2
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["CAP", "non-CAP"])
    ax.set_ylabel("Mortality Proportion")

    ax.set_ylim([0, 1])
    ax.set_xlim([-0.5, 1.5])

    return fig
