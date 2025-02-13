"""Figure J1: Ramping Mortality Risk"""

import numpy as np
from anndata import read_h5ad

from pf2.data_import import condition_factors_meta
from pf2.figures.common import getSetup
from pf2.predict import predict_mortality

PATIENTS = [5429, 5469, 7048]


def makeFigure():
    data = read_h5ad("/opt/northwest_bal/full_fitted.h5ad", backed="r")
    cond_fact_meta_df = condition_factors_meta(data)

    probabilities, labels, (c_plsr, nc_plsr) = predict_mortality(
        data, cond_fact_meta_df, proba=True
    )
    cond_fact_meta_df = cond_fact_meta_df.loc[probabilities.index, :]
    cond_fact_meta_df = cond_fact_meta_df.sort_values(
        "ICU Day",
        ascending=True
    )
    cond_fact_meta_df.iloc[:, :50] /= abs(cond_fact_meta_df.iloc[
        :,
        :50
    ]).max(axis=0)

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
        samples = (cond_fact_meta_df.loc[
            cond_fact_meta_df.loc[:, "patient_id"] == patient_id,
            :
        ]).index
        ax.plot(
            cond_fact_meta_df.loc[samples, "ICU Day"],
            probabilities.loc[samples]
        )
        x_pos = cond_fact_meta_df.loc[samples, "ICU Day"].max() + 1
        y_pos = probabilities.loc[samples].iloc[-1]
        ax.text(
            x_pos,
            y_pos,
            s=str(patient_id),
            ha="left",
            ma="left",
            va="center"
        )

    ax.set_xlim((0, 100))
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
            s=component,
        )

    ax.plot([-100, 100], [0, 0], linestyle="--", color="k", zorder=-3)
    ax.plot([0, 0], [-100, 100], linestyle="--", color="k", zorder=-3)
    ax.set_xlim((-0.4, 0.4))
    ax.set_ylim((-0.4, 0.4))

    ax.set_xlabel("COVID")
    ax.set_ylabel("Non-COVID")
    ax.set_title("PLSR Scores")

    cond_fact_meta_df = cond_fact_meta_df.loc[
        cond_fact_meta_df.loc[:, "patient_id"].duplicated(keep=False)
    ]

    for column_index, comp_set in enumerate([protective, deviant]):
        for row_index, comp in enumerate(comp_set):
            ax = axs[column_index + row_index * 2 + 2]
            for multi_id in cond_fact_meta_df.loc[:, "patient_id"].unique():
                if multi_id not in PATIENTS:
                    samples = (cond_fact_meta_df.loc[
                        cond_fact_meta_df.loc[
                            :,
                            "patient_id"
                        ] == multi_id,
                        :
                    ]).index
                    ax.plot(
                        np.linspace(0, 1, len(samples)),
                        cond_fact_meta_df.loc[
                            samples,
                            f"Cmp. {comp}"
                        ],
                        color="grey",
                        alpha=0.25,
                    )

            for patient_id in PATIENTS:
                samples = (cond_fact_meta_df.loc[
                    cond_fact_meta_df.loc[
                        :,
                        "patient_id"
                    ] == patient_id,
                    :
               ]).index
                ax.plot(
                    np.linspace(0, 1, len(samples)),
                    cond_fact_meta_df.loc[
                        samples,
                        f"Cmp. {comp}"
                    ],
                    label=patient_id,
                )

            ax.legend()
            ax.set_title(f"Component {comp}")
            ax.set_ylabel("Patient Factor")
            ax.set_xlabel("Time")

            ax.set_xticks([])
            ax.set_yticks([0, 1])
            ax.set_ylim((-0.1, 1.1))

    return fig
