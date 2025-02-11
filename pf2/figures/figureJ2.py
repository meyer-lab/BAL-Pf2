"""Figure J2: Early/Late ROC Curves"""

import numpy as np
import pandas as pd
from anndata import read_h5ad
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from pf2.data_import import convert_to_patients, import_meta
from pf2.figures.common import getSetup
from pf2.predict import predict_mortality


def makeFigure():
    data = read_h5ad(
        "/opt/northwest_bal/full_fitted.h5ad", backed="r"
    )
    meta = import_meta(drop_duplicates=False)
    meta.set_index("sample_id", inplace=True)
    conversions = convert_to_patients(data, sample=True)

    patient_factor = pd.DataFrame(
        data.uns["Pf2_A"],
        index=conversions,
        columns=np.arange(data.uns["Pf2_A"].shape[1]) + 1,
    )

    shared_indices = patient_factor.index.intersection(meta.index)
    patient_factor = patient_factor.loc[shared_indices, :]
    meta = meta.loc[shared_indices, :]

    axs, fig = getSetup((9, 3), (1, 3))

    probabilities, labels, _ = predict_mortality(
        patient_factor,
        meta,
        proba=True
    )
    meta = meta.loc[probabilities.index, :]
    predicted = probabilities.round().astype(int)

    w1_patients = meta.loc[meta.loc[:, "icu_day"] < 7, :].index
    m1_patients = meta.loc[meta.loc[:, "icu_day"].between(8, 27), :].index
    late_patients = meta.loc[meta.loc[:, "icu_day"] > 28, :].index
    names = [
        f"ICU Week 1 ({len(w1_patients)})",
        f"ICU Month 1 ({len(m1_patients)})",
        f"ICU Past Month 1 ({len(late_patients)})"
    ]
    patient_sets = [w1_patients, m1_patients, late_patients]
    for name, patient_set, ax in zip(names, patient_sets, axs):
        patient_pred = predicted.loc[patient_set]
        patient_proba = probabilities.loc[patient_set]
        patient_labels = labels.loc[patient_set]
        acc = accuracy_score(patient_labels, patient_pred)
        auc_roc = roc_auc_score(patient_labels, patient_proba)
        fpr, tpr, _ = roc_curve(patient_labels, patient_proba)

        ax.plot([0, 1], [0, 1], linestyle="--", color="k")
        ax.plot(fpr, tpr)
        ax.text(
            0.99,
            0.01,
            s=f"AUC ROC: {round(auc_roc, 2)}\nAccuracy: {round(acc, 2)}",
            ha="right",
            va="bottom",
            transform=ax.transAxes,
        )

        ax.set_title(name)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    return fig
