"""Figure J2: Early/Late ROC Curves"""

from anndata import read_h5ad
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from pf2.data_import import condition_factors_meta
from pf2.figures.common import getSetup
from pf2.predict import predict_mortality


def makeFigure():
    data = read_h5ad("/opt/northwest_bal/full_fitted.h5ad", backed="r")
    cond_fact_meta_df = condition_factors_meta(data)

    probabilities, labels, _ = predict_mortality(
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

    axs, fig = getSetup((9, 3), (1, 3))

    w1_patients = cond_fact_meta_df.loc[
        cond_fact_meta_df.loc[:, "ICU Day"] < 7,
        :
    ].index
    m1_patients = cond_fact_meta_df.loc[
        cond_fact_meta_df.loc[:, "ICU Day"].between(8, 27),
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
    for name, patient_set, ax in zip(names, patient_sets, axs):
        patient_pred = probabilities.loc[patient_set].round().astype(int)
        patient_proba = probabilities.loc[patient_set]  # type: ignore
        patient_labels = labels.loc[patient_set]
        acc = accuracy_score(patient_labels, patient_pred)
        auc_roc = roc_auc_score(patient_labels, patient_proba)
        fpr, tpr, _ = roc_curve(patient_labels, patient_proba)

        ax.plot([0, 1], [0, 1], linestyle="--", color="k")
        ax.plot(fpr, tpr)
        ax.text(
            0.99,
            0.01,
            s=f"AUC ROC: {round(auc_roc, 2)}\nAccuracy: {round(acc, 2)}",  # type: ignore
            ha="right",
            va="bottom",
            transform=ax.transAxes,
        )

        ax.set_title(name)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim((0, 1))
        ax.set_ylim((0, 1))

    return fig
