"""Figure J7a: Clinical Correlates Heatmap"""

import numpy as np
import pandas as pd
import seaborn as sns
from anndata import read_h5ad
from sklearn.preprocessing import LabelEncoder
from scipy.stats import f_oneway, pearsonr, ttest_ind, tukey_hsd

from pf2.data_import import convert_to_patients, import_meta
from pf2.figures.common import getSetup
from pf2.utils import reorder_table

COMPONENTS = np.array([14, 23, 16, 10, 34, 3, 15, 67, 55, 47, 4, 1, 22, 62])
TTEST = [
    "ecmo_flag",
    "episode_category",
    "episode_etiology",
    "pathogen_virus_detected",
    "pathogen_bacteria_detected",
    "pathogen_fungi_detected",
    "smoking_status",
    "icu_stay",
    "admission_source_name",
    "global_cause_failure",
    "patient_category",
    "covid_status",
    "gender",
    "tracheostomy_flag",
    "immunocompromised_flag",
    "norepinephrine_flag",
    "remdesivir_received",
    "episode_is_cured",
]
CORRELATES = [
    "age",
    "bmi",
    "number_of_icu_stays",
    "cumulative_icu_days",
    "icu_day",
    "admit_sofa_score",
    "admit_aps_score",
    "cumulative_intubation_days",
    "BAL_amylase",
    "BAL_pct_neutrophils",
    "BAL_pct_macrophages",
    "BAL_pct_monocytes",
    "BAL_pct_lymphocytes",
    "BAL_pct_eosinophils",
    "BAL_pct_other",
    "temperature",
    "heart_rate",
    "systolic_blood_pressure",
    "diastolic_blood_pressure",
    "mean_arterial_pressure",
    "norepinephrine_rate",
    "respiratory_rate",
    "oxygen_saturation",
    "rass_score",
    "peep",
    "fio2",
    "plateau_pressure",
    "lung_compliance",
    "minute_ventilation",
    "abg_ph",
    "abg_paco2",
    "pao2fio2_ratio",
    "wbc_count",
    "bicarbonate",
    "creatinine",
    "albumin",
    "bilirubin",
    "crp",
    "d_dimer",
    "ferritin",
    "ldh",
    "lactic_acid",
    "procalcitonin",
    "nat_score",
    "steroid_dose",
    "episode_duration",
]
BINNED = [
    ("icu_day", 2),
    ("number_of_icu_stays", 2),
    ("gcs_eye_opening", 2),
    ("gcs_motor_response", 2),
]
ENCODER = LabelEncoder()


def makeFigure():
    meta = import_meta(drop_duplicates=False)
    data = read_h5ad("/opt/northwest_bal/full_fitted.h5ad", backed="r")
    conversions = convert_to_patients(data, sample=True)

    patient_factor = pd.DataFrame(
        data.uns["Pf2_A"],
        index=conversions,
        columns=np.arange(data.uns["Pf2_A"].shape[1]) + 1,
    )
    patient_factor /= abs(patient_factor).max(axis=0)

    meta = meta.set_index("sample_id")
    shared_indices = patient_factor.index.intersection(meta.index)
    patient_factor = patient_factor.loc[shared_indices, :]
    meta = meta.loc[shared_indices, :]

    ttest_p_values = pd.DataFrame(columns=patient_factor.columns, dtype=float)
    t_stats = ttest_p_values.copy(deep=True)
    anova_p_values = ttest_p_values.copy(deep=True)
    anova_stats = ttest_p_values.copy(deep=True)
    tukey_p_values = ttest_p_values.copy(deep=True)
    tukey_stats = ttest_p_values.copy(deep=True)
    for column in TTEST:
        _patient_factor = patient_factor.loc[~meta.loc[:, column].isna(), :]
        _meta = meta.loc[~meta.loc[:, column].isna(), :]
        ENCODER.fit(_meta.loc[:, column])
        values = ENCODER.classes_
        if column == "covid_status":
            print()

        if len(values) == 2:
            for comp in patient_factor.columns:
                result = ttest_ind(
                    _patient_factor.loc[_meta.loc[:, column] == values[1], comp],
                    _patient_factor.loc[_meta.loc[:, column] == values[0], comp],
                )
                ttest_p_values.loc[column, comp] = result.pvalue
                t_stats.loc[column, comp] = result.statistic
        else:
            if any(
                [
                    _patient_factor.loc[_meta.loc[:, column] == value, :].shape[0] < 10
                    for value in values
                ]
            ):
                continue
            groups = [
                _patient_factor.loc[_meta.loc[:, column] == value, :]
                for value in values
            ]
            result = f_oneway(*groups, axis=0)
            anova_p_values.loc[column, :] = result.pvalue
            anova_stats.loc[column, :] = result.statistic
            for comp in patient_factor.columns:
                _groups = [group.loc[:, comp] for group in groups]
                result = tukey_hsd(*_groups)
                for index_1, value_1 in enumerate(values):
                    for index_2, value_2 in enumerate(values[index_1 + 1:]):
                        index_2 += 1
                        tukey_stats.loc[f"{column}: {value_1} - {value_2}", comp] = (
                            result.statistic[index_1, index_2]
                        )
                        tukey_p_values.loc[f"{column}: {value_1} - {value_2}", comp] = (
                            result.pvalue[index_1, index_2]
                        )

    for binned_var in BINNED:
        (column, threshold) = binned_var
        _patient_factor = patient_factor.loc[~meta.loc[:, column].isna(), :]
        _meta = meta.loc[~meta.loc[:, column].isna(), :]
        for comp in patient_factor.columns:
            result = ttest_ind(
                _patient_factor.loc[_meta.loc[:, column] >= threshold, comp],
                _patient_factor.loc[_meta.loc[:, column] < threshold, comp],
            )
            ttest_p_values.loc[f"{column} >= {threshold}", comp] = result.pvalue
            t_stats.loc[f"{column} >= {threshold}", comp] = result.statistic

    corr_p = pd.DataFrame(index=CORRELATES, columns=patient_factor.columns, dtype=float)
    corr_coef = corr_p.copy(deep=True)
    for column in CORRELATES:
        _patient_factor = patient_factor.loc[~meta.loc[:, column].isna(), :]
        values = meta.loc[:, column].dropna()
        for comp in patient_factor.columns:
            result = pearsonr(_patient_factor.loc[:, comp], values)
            corr_p.loc[column, comp] = result.pvalue
            corr_coef.loc[column, comp] = result.statistic

    axs, fig = getSetup(
        (6, 12),
        (4, 1),
        gs_kws={
            "height_ratios": [
                corr_coef.shape[0],
                t_stats.shape[0],
                anova_stats.shape[0],
                tukey_stats.shape[0],
            ]
        },
    )
    merged = pd.concat([corr_coef, t_stats])
    merged_order = reorder_table(merged.T)

    for stat, p_vals, label, ax in zip(
        (corr_coef, t_stats, anova_stats, tukey_stats),
        (corr_p, ttest_p_values, anova_p_values, tukey_p_values),
        ["Pearson Correlation", "T-Statistic", "ANOVA", "Tukey HSD"],
        axs,
    ):
        # stat = stat.iloc[:, merged_order].loc[:, COMPONENTS]
        # p_vals = p_vals.iloc[:, merged_order].loc[:, COMPONENTS]

        annot = np.empty(p_vals.shape, dtype=np.dtype("U100"))
        annot[p_vals < 0.05] = "*"
        annot[p_vals < 0.01] = "**"
        annot[p_vals < 0.001] = "***"

        sns.heatmap(
            stat,
            annot=annot,
            cmap="coolwarm",
            center=0,
            fmt="s",
            ax=ax,
            annot_kws={"ha": "center", "ma": "center", "va": "center"},
            cbar_kws={"label": label, "shrink": 8 / stat.shape[0]},
            xticklabels=ax == axs[-1],
            yticklabels=True,
        )

        ax.set_ylabel("Clinical Variable")
        if ax == axs[-1]:
            ax.set_xlabel("PF2 Component")

    return fig
