"""Figure J3d_e: Early/Late ROC Curves"""
from os.path import join

import pandas as pd
from anndata import read_h5ad
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from pf2.data_import import NO_META_SAMPLES, condition_factors_meta
from pf2.figures.common import getSetup
from pf2.predict import predict_mortality

DATA_PATH = join("/opt", "northwest_bal")


def makeFigure():
    data = read_h5ad("/opt/northwest_bal/full_fitted.h5ad", backed="r")
    cond_fact_meta_df = condition_factors_meta(data)

    meta = pd.read_csv(join(DATA_PATH, "04_external.csv"), index_col=0)
    meta.sort_values("icu_day", inplace=True)

    patient_map = meta.set_index("sample_id")
    patient_map = patient_map.loc[~patient_map.index.duplicated()]
    patient_map = patient_map.loc[:, "patient_id"]
    for sample_id, patient_id in NO_META_SAMPLES.items():
        patient_map.loc[sample_id] = patient_id

    meta = meta.loc[
        ~meta.loc[:, "patient_id"].duplicated(keep="last"),
        :
    ].set_index("patient_id")
    last_tp = meta.loc[:, "icu_day"]

    probabilities, labels, _ = predict_mortality(
        data, cond_fact_meta_df, proba=True
    )
    cond_fact_meta_df = cond_fact_meta_df.loc[probabilities.index, :]
    cond_fact_meta_df = cond_fact_meta_df.sort_values(
        "ICU Day",
        ascending=True
    )
    cond_fact_meta_df.iloc[:, :80] /= abs(cond_fact_meta_df.iloc[
        :,
        :80
    ]).max(axis=0)

    axs, fig = getSetup((6, 3), (1, 2))

    w1_patients = cond_fact_meta_df.loc[
        cond_fact_meta_df.loc[:, "ICU Day"] < 8,
        :
    ].index
    m1_patients = cond_fact_meta_df.loc[
        cond_fact_meta_df.loc[:, "ICU Day"].between(8, 28),
        :
    ].index
    late_patients = cond_fact_meta_df.loc[
        cond_fact_meta_df.loc[:, "ICU Day"] > 28,
        :
    ].index
    names = [
        f"ICU Week 1 ({len(w1_patients)})",
        f"ICU Month 1 ({len(m1_patients)})",
        f"ICU Past Month 1 ({len(late_patients)})",
    ]
    patient_sets = [w1_patients, m1_patients, late_patients]
    performance_text = ""
    ax = axs[0]
    for name, patient_set in zip(names, patient_sets, strict=False):
        patient_pred = probabilities.loc[patient_set].round().astype(int)
        patient_proba = probabilities.loc[patient_set]  # type: ignore
        patient_labels = labels.loc[patient_set]
        acc = accuracy_score(patient_labels, patient_pred)
        auc_roc = roc_auc_score(patient_labels, patient_proba)
        fpr, tpr, _ = roc_curve(patient_labels, patient_proba)

        ax.plot([0, 1], [0, 1], linestyle="--", color="k")
        ax.plot(fpr, tpr)
        performance_text += f"{name} (AUC: {round(auc_roc, 2)}"  # type: ignore
        performance_text += f", ACC: {round(acc, 2)})\n"  # type: ignore

    ax.set_title("Performance by Sample Date")
    axs[0].text(
        0.99,
        0.01,
        s=performance_text,  # type: ignore
        ha="right",
        va="bottom",
        transform=axs[0].transAxes
    )

    last_tp = patient_map.loc[probabilities.index].replace(last_tp)
    time_to_last = last_tp - cond_fact_meta_df.loc[:, "ICU Day"]

    w1_ttd = time_to_last.loc[time_to_last < 8].index
    m1_ttd = time_to_last.loc[time_to_last.between(8, 28)].index
    long_ttd = time_to_last.loc[time_to_last > 28].index
    ax = axs[1]

    names = [
        f"<1 Weeks ({len(w1_ttd)})",
        f"<4 Weeks ({len(m1_ttd)})",
        f">4 Weeks ({len(long_ttd)})",
    ]
    patient_sets = [w1_ttd, m1_ttd, long_ttd]
    performance_text = ""
    for name, patient_set in zip(names, patient_sets, strict=False):
        patient_pred = probabilities.loc[patient_set].round().astype(int)
        patient_proba = probabilities.loc[patient_set]  # type: ignore
        patient_labels = labels.loc[patient_set]
        acc = accuracy_score(patient_labels, patient_pred)
        auc_roc = roc_auc_score(patient_labels, patient_proba)
        fpr, tpr, _ = roc_curve(patient_labels, patient_proba)

        ax.plot([0, 1], [0, 1], linestyle="--", color="k")
        ax.plot(fpr, tpr)
        performance_text += f"{name} (AUC: {round(auc_roc, 2)}"  # type: ignore
        performance_text += f", ACC: {round(acc, 2)})\n"  # type: ignore

    ax.set_title("Performance by Time-to-Discharge")
    ax.text(
        0.99,
        0.01,
        s=performance_text,  # type: ignore
        ha="right",
        va="bottom",
        transform=ax.transAxes
    )

    for ax in axs:
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))

    return fig
