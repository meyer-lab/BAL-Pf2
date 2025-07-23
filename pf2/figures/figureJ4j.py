"""
Figure J4j: CD8+ & T-reg Balance
"""

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import find
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC

from ..data_import import add_obs
from .common import getSetup

COMPONENTS = [3, 15]


def run_svc(data, labels, ax):
    skf = StratifiedKFold(n_splits=10)
    svm = SVC(kernel="linear", probability=True)
    print(np.mean(cross_val_score(svm, data, labels, cv=skf)))

    svm.fit(data, labels)
    xx, yy = np.meshgrid(
        np.linspace(0, 4, 21),
        np.linspace(0, 4, 21)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    prob_map = svm.predict_proba(grid)[:, 1].reshape(xx.shape)

    contour = ax.contourf(
        xx,
        yy,
        prob_map,
        cmap='coolwarm',
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
    add_obs(data, "episode_etiology")

    data.obs.loc[:, "covid_status"] = data.obs.loc[:, "covid_status"].fillna(
        False
    )
    data = data[data.obs.loc[:, "covid_status"].astype(bool), :]
    data = data[data.obs.loc[:, "icu_day"] > 7, :]

    tc_proportions = pd.DataFrame(
        index=data.obs.loc[:, "sample_id"].unique(),
        columns=["FOXP3", "ITGA1", "binary_outcome"]
    )
    for sample_id in data.obs.loc[:, "sample_id"].unique():
        sample_data = data[data.obs.loc[:, "sample_id"] == sample_id, :]
        data = data[data.obs.loc[:, "sample_id"] != sample_id, :]
        tc_proportions.loc[sample_id, "FOXP3"] = np.log10(1 + len(
            set(find(sample_data[:, "FOXP3"].X)[0])
        ))
        tc_proportions.loc[sample_id, "ITGA1"] = np.log10(1 + len(
            set(find(sample_data[:, "ITGA1"].X)[0])
        ))
        tc_proportions.loc[sample_id, "binary_outcome"] = sample_data.obs.loc[
            :,
            "binary_outcome"
        ].iloc[0]

    run_svc(
        tc_proportions.iloc[:, :-1],
        tc_proportions.iloc[:, -1].values.astype(int),
        ax
    )

    ax.scatter(
        tc_proportions.loc[:, "FOXP3"],
        tc_proportions.loc[:, "ITGA1"],
        c=tc_proportions.loc[:, "binary_outcome"].replace({
            0: "darkblue",
            1: "darkred"
        })
    )
    ax.set_xlabel("FOXP3+ Proportion")
    ax.set_ylabel("ITGA1+ Proportion")

    return f
