"""Figure J1: Ramping Mortality Risk"""

import numpy as np
import pandas as pd
from anndata import read_h5ad

from pf2.data_import import convert_to_patients, import_meta
from pf2.figures.common import getSetup
from pf2.predict import predict_mortality

PATIENTS = [5429, 5469, 7048]


def makeFigure():
    meta = import_meta(drop_duplicates=False)
    data = read_h5ad("/opt/northwest_bal/full_fitted.h5ad", backed="r")
    sample_conversions = convert_to_patients(data, sample=True)

    patient_factor = pd.DataFrame(
        data.uns["Pf2_A"],
        index=sample_conversions,
        columns=np.arange(data.uns["Pf2_A"].shape[1]) + 1,
    )
    patient_factor /= abs(patient_factor).max(axis=0)

    meta = meta.set_index("sample_id")
    shared_indices = patient_factor.index.intersection(meta.index)
    patient_factor = patient_factor.loc[shared_indices, :]
    meta = meta.loc[shared_indices, :]

    probabilities, labels, (c_plsr, nc_plsr) = predict_mortality(
        patient_factor,
        meta,
        proba=True
    )
    components = np.argsort(c_plsr.x_loadings_[:, 0])
    protective = components[:3] + 1
    deviant = components[-3:] + 1
    deviant = deviant[::-1]

    axs, fig = getSetup(
        (8, 4 * 2),
        (4, 2),
        gs_kws={"height_ratios": [2] + [1] * 3}
    )

    ax = axs[0]
    for patient_id in PATIENTS:
        _meta = meta.loc[meta.loc[:, "patient_id"] == patient_id, :]
        _meta = _meta.sort_values("icu_day", ascending=True)
        _probabilities = probabilities.loc[_meta.index]
        _labels = labels.loc[_meta.index]

        ax.plot(
            _meta.loc[:, "icu_day"],
            _probabilities
        )
        x_pos = _meta.loc[:, "icu_day"].max() + 1
        y_pos = _probabilities.iloc[-1]
        if patient_id == 6308:
            x_pos -= 1
            y_pos += 0.025
        ax.text(
            x_pos,
            y_pos,
            s=patient_id,
            ha="left",
            ma="left",
            va="center"
        )

    ax.set_xlim([0, 100])
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_ylabel("Mortality Probability")
    ax.set_xlabel("ICU Day")
    ax.set_title("Patient Mortality Risk")

    ax = axs[1]
    ax.scatter(
        c_plsr.y_loadings_[0, 0],
        nc_plsr.y_loadings_[0, 0],
        s=150,
        c="tab:red"
    )
    ax.scatter(
        c_plsr.x_loadings_[:, 0],
        nc_plsr.x_loadings_[:, 0],
        s=120,
        facecolors="white",
        edgecolors="k",
    )
    for index, component in enumerate(c_plsr.coef_.index):
        ax.text(
            c_plsr.x_loadings_[index, 0],
            nc_plsr.x_loadings_[index, 0] - 0.001,
            ha="center",
            ma="center",
            va="center",
            s=component
        )

    ax.plot(
        [-100, 100],
        [0, 0],
        linestyle="--",
        color="k",
        zorder=-3
    )
    ax.plot(
        [0, 0],
        [-100, 100],
        linestyle="--",
        color="k",
        zorder=-3
    )
    ax.set_xlim([-0.4, 0.4])
    ax.set_ylim([-0.4, 0.4])

    ax.set_xlabel("COVID")
    ax.set_ylabel("Non-COVID")
    ax.set_title("PLSR Scores")

    meta = meta.loc[probabilities.index, :]
    meta = meta.loc[meta.loc[:, "patient_id"].duplicated(keep=False), :]

    for column_index, comp_set in enumerate([protective, deviant]):
        for row_index, comp in enumerate(comp_set):
            ax = axs[column_index + row_index * 2 + 2]
            for multi_id in meta.loc[:, "patient_id"].unique():
                if multi_id not in PATIENTS:
                    _meta = meta.loc[meta.loc[:, "patient_id"] == multi_id, :]
                    _meta = _meta.sort_values("icu_day", ascending=True)
                    ax.plot(
                        np.linspace(0, 1, _meta.shape[0]),
                        patient_factor.loc[_meta.index, comp],
                        color="grey",
                        alpha=0.25
                    )

            for patient_id in PATIENTS:
                _meta = meta.loc[meta.loc[:, "patient_id"] == patient_id, :]
                _meta = _meta.sort_values("icu_day", ascending=True)
                ax.plot(
                    np.linspace(0, 1, _meta.shape[0]),
                    patient_factor.loc[_meta.index, comp],
                    label=patient_id
                )

            ax.legend()
            ax.set_title(f"Component {comp}")
            ax.set_ylabel("Patient Factor")
            ax.set_xlabel("Time")

            ax.set_xticks([])
            ax.set_yticks([0, 1])
            ax.set_ylim([-0.1, 1.1])

    return fig
