"""
Figure J4i: Macrophage-Granulocyte Axis
"""

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import find
from sklearn.svm import SVR

from .common import getSetup
from ..data_import import add_obs

COMPONENTS = [3, 15]


def run_svc(data, labels, ax):
    svm = SVR(kernel="poly")

    svm.fit(data, labels)
    print(svm.score(data, labels))
    xx, yy = np.meshgrid(
        np.linspace(0, 3, 31),
        np.linspace(0, 3, 31)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    prob_map = svm.predict(grid).reshape(xx.shape)
    prob_map = np.clip(prob_map, 0, 100)

    contour = ax.contourf(
        xx,
        yy,
        prob_map,
        cmap='Purples',
        linestyles='--'
    )
    plt.colorbar(contour, ax=ax)


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    axs, f = getSetup(
        (4.5, 4),
        (1, 1)
    )
    ax = axs[0]

    data = anndata.read_h5ad(
        "/opt/northwest_bal/full_fitted.h5ad"
    )

    add_obs(data, "binary_outcome")
    add_obs(data, "covid_status")
    add_obs(data, "icu_day")
    add_obs(data, "bmi")
    add_obs(data, "cumulative_icu_days")
    add_obs(data, "pathogen_bacteria_detected")
    add_obs(data, "BAL_pct_neutrophils")

    data = data[data.obs.loc[:, "covid_status"] == True, :]
    data = data[data.obs.loc[:, "icu_day"] > 7, :]

    tc_proportions = pd.DataFrame(
        index=data.obs.loc[:, "sample_id"].unique(),
        columns=["ADORA3", "PADI4", "n_pct"]
    )
    for sample_id in data.obs.loc[:, "sample_id"].unique():
        sample_data = data[data.obs.loc[:, "sample_id"] == sample_id, :]
        data = data[data.obs.loc[:, "sample_id"] != sample_id, :]
        tc_proportions.loc[sample_id, "ADORA3"] = np.log10(1 + len(
            set(find(sample_data[:, "ADORA3"].X)[0])
        ))
        tc_proportions.loc[sample_id, "PADI4"] = np.log10(1 + len(
            set(find(sample_data[:, "PADI4"].X)[0])
        ))
        tc_proportions.loc[sample_id, "n_pct"] = sample_data.obs.loc[
            :,
            "BAL_pct_neutrophils"
        ].iloc[0]

    tc_proportions = tc_proportions.loc[
        ~tc_proportions.iloc[:, -1].isna(),
        :
    ]

    run_svc(
        tc_proportions.iloc[:, :-1],
        tc_proportions.iloc[:, -1],
        ax
    )

    ax.scatter(
        tc_proportions.loc[:, "ADORA3"],
        tc_proportions.loc[:, "PADI4"],
        c=tc_proportions.loc[:, "n_pct"],
        cmap="Purples",
        edgecolors="black"
    )
    ax.set_xlabel("ADORA3+ Proportion")
    ax.set_ylabel("PADI4+ Proportion")

    return f
