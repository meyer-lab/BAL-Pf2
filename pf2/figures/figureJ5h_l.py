"""
Figure J4l: Evolving Treg/CD8 Behaviors
"""

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import find
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC

from .common import getSetup
from ..data_import import add_obs

PATIENTS = [424, 3152, 5469, 6308, 7048]


def run_svc(data, labels):
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

    return xx, yy, prob_map


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    axs, f = getSetup(
        (4 * len(PATIENTS) + 0.5, 4),
        (1, len(PATIENTS))
    )
    ax = axs[0]

    data = anndata.read_h5ad(
        "/opt/northwest_bal/full_fitted.h5ad"
    )

    add_obs(data, "binary_outcome")
    add_obs(data, "covid_status")
    add_obs(data, "icu_day")
    add_obs(data, "episode_etiology")

    data = data[data.obs.loc[:, "covid_status"] == True, :]
    tc_proportions = pd.DataFrame(
        index=data.obs.loc[:, "sample_id"].unique(),
        columns=["FOXP3", "ITGA1", "icu_day", "patient_id", "binary_outcome"]
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
        tc_proportions.loc[sample_id, "icu_day"] = sample_data.obs.loc[
            :,
            "icu_day"
        ].iloc[0]
        tc_proportions.loc[sample_id, "patient_id"] = sample_data.obs.loc[
            :,
            "patient_id"
        ].iloc[0]

    svc_data = tc_proportions.loc[
        tc_proportions.loc[:, "icu_day"] > 7,
        :
    ]
    contour_params = run_svc(
        svc_data.loc[:, ["FOXP3", "ITGA1"]],
        svc_data.loc[:, "binary_outcome"].values.astype(int)
    )

    for ax, patient in zip(axs, PATIENTS):
        patient_samples = tc_proportions.loc[
            tc_proportions.loc[:, "patient_id"] == patient,
            :
        ]
        patient_samples = patient_samples.sort_values("icu_day")
        contour = ax.contourf(
            *contour_params,
            cmap='coolwarm',
            linestyles='--'
        )
        ax.scatter(
            tc_proportions.loc[:, "FOXP3"],
            tc_proportions.loc[:, "ITGA1"],
            c="black",
            alpha=0.5
        )
        ax.plot(
            patient_samples.loc[:, "FOXP3"],
            patient_samples.loc[:, "ITGA1"],
            linewidth=2,
            color="black"
        )
        ax.scatter(
            patient_samples.loc[:, "FOXP3"],
            patient_samples.loc[:, "ITGA1"],
            s=patient_samples.loc[:, "icu_day"].astype(float),
            color="lightgrey",
            alpha=0.5
        )
        ax.set_title(patient)
        ax.set_xlabel("FOXP3+ Proportion")
        ax.set_ylabel("ITGA1+ Proportion")
        if ax == axs[-1]:
            plt.colorbar(contour)

    return f
