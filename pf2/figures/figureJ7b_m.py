"""Figure J7b_m: Clinical Correlates -- Detail Plots"""

from decimal import Decimal

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, ttest_ind
import seaborn as sns
import statsmodels.api as sm
from anndata import read_h5ad
from sklearn.preprocessing import LabelEncoder

from pf2.data_import import convert_to_patients, import_meta
from pf2.figures.common import getSetup

COMPONENTS = [14, 23, 16, 10, 34, 3, 15, 67, 55, 47, 4, 1, 22, 62]
T_TEST = [
    "multiple_etiology",
    "covid_status",
    "immunocompromised_flag",
    "pathogen_bacteria_detected",
    "pathogen_fungi_detected",
    "pathogen_virus_detected"
]
ANOVA = [
    "episode_etiology",
    "episode_category"
]
PEARSON = [
    "BAL_pct_lymphocytes",
    "BAL_pct_neutrophils",
    "BAL_pct_macrophages",
    "lymphocyte_count",
    "neutrophil_count",
    "macrophage_count",
    "peep",
    "lactic_acid",
    "cumulative_intubation_days",
    "cumulative_icu_days",
    "icu_day",
    "bmi"
]
N_ROWS = len(T_TEST) + len(ANOVA) + len(PEARSON)
ENCODER = LabelEncoder()


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

    cell_counts = data.obs.loc[
        :,
        "sample_id"
    ].value_counts() / meta.loc[:, "BAL_pct_macrophages":].sum(axis=1)
    meta.loc[:, "lymphocyte_count"] = np.log(
        meta.loc[
            :,
            "BAL_pct_lymphocytes"
        ] * cell_counts + 1
    )
    meta.loc[:, "neutrophil_count"] = np.log(
        meta.loc[
            :,
            "BAL_pct_neutrophils"
        ] * cell_counts + 1
    )
    meta.loc[:, "macrophage_count"] = np.log(
        meta.loc[
            :,
            "BAL_pct_macrophages"
        ] * cell_counts + 1
    )

    meta = meta.sort_values(["patient_id", "icu_day"], ascending=True)
    meta.loc[:, "multiple_etiology"] = False
    for patient in meta.loc[
        meta.loc[:, "covid_status"].fillna(False),
        "patient_id"
    ].unique():
        patient_meta = meta.loc[meta.loc[:, "patient_id"] == patient, :]
        etio_col = patient_meta.loc[
            :, ["pathogen_bacteria_detected", "pathogen_fungi_detected"]
        ].any(axis=1)
        if etio_col.any():
            etio_col.loc[etio_col.idxmax() :] = True

        meta.loc[patient_meta.index, "multiple_etiology"] = etio_col

    patient_factor = patient_factor.loc[meta.index, :]
    axs, fig = getSetup(
        (3 * len(COMPONENTS), 3 * N_ROWS),
        (N_ROWS, len(COMPONENTS))
    )

    for row, variable in enumerate(T_TEST + ANOVA):
        for col, component in enumerate(COMPONENTS):
            ax = axs[row * len(COMPONENTS) + col]
            meta_col = meta.loc[:, variable].dropna()
            _patient_factor = patient_factor.loc[meta_col.index, :]
            _patient_factor.loc[:, variable] = ENCODER.fit_transform(meta_col)
            for index, value in enumerate(ENCODER.classes_):
                ax.errorbar(
                    index,
                    _patient_factor.loc[
                        _patient_factor.loc[:, variable] == index, component
                    ].mean(),
                    yerr=_patient_factor.loc[
                        _patient_factor.loc[:, variable] == index, component
                    ].std(),
                    capsize=10,
                    markersize=20,
                    linewidth=3,
                    markeredgewidth=3,
                    marker="_",
                    color="k",
                    zorder=10,
                )

            sns.stripplot(
                _patient_factor,
                x=variable,
                y=component,
                ax=ax
            )
            ax.set_xticks(np.arange(len(ENCODER.classes_)))
            ax.set_xticklabels(ENCODER.classes_)

            if len(ENCODER.classes_) == 2:
                result = ttest_ind(
                    _patient_factor.loc[
                        _patient_factor.loc[:, variable] == 1,
                        component
                    ],
                    _patient_factor.loc[
                        _patient_factor.loc[:, variable] == 0,
                        component
                    ],
                )
            else:
                groups = [
                    _patient_factor.loc[
                        _patient_factor.loc[:, variable] == index,
                        component
                    ]
                    for index in np.arange(len(ENCODER.classes_))
                ]
                result = f_oneway(*groups, axis=0)

            ax.text(
                0.98,
                0.02,
                s=f"p-value: {Decimal(result.pvalue):.2E}",
                ha="right",
                ma="right",
                va="bottom",
                transform=ax.transAxes,
            )

            ax.set_ylim([-0.05, 1.05])

    for row, variable in enumerate(PEARSON):
        for col, component in enumerate(COMPONENTS):
            ax = axs[
                len(T_TEST + ANOVA) * len(COMPONENTS) +
                len(COMPONENTS) * row +
                col
            ]
            sm_data = patient_factor.loc[:, component]
            sm_data = sm.add_constant(sm_data)

            # if variable.startswith("BAL") or variable.endswith("days"):
            #     meta.loc[:, variable] = np.log(meta.loc[:, variable])

            model = sm.OLS(meta.loc[:, variable], sm_data, missing="drop")
            results = model.fit()
            ax.scatter(
                patient_factor.loc[:, component],
                meta.loc[:, variable],
                s=6
            )

            x_lim = ax.get_xlim()
            y_lim = ax.get_ylim()

            xs = [0, patient_factor.loc[:, component].max() * 1.05]
            ys = [
                results.params.iloc[0] + results.params.iloc[1] * xs[0],
                results.params.iloc[0] + results.params.iloc[1] * xs[1],
            ]

            ax.plot(xs, ys, color="k", linestyle="--")
            ax.set_xlabel(component)  # type: ignore
            ax.set_ylabel(variable)  # type: ignore

            ax.text(
                0.98,
                0.02,
                s=f"R2: {round(results.rsquared_adj, 3)}\n"
                f"p-value: {Decimal(results.pvalues.loc[component]):.2E}",
                ha="right",
                ma="right",
                va="bottom",
                transform=ax.transAxes,
            )
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)

    return fig
