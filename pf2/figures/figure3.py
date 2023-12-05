"""Figure 3: ROC Curves"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from pf2.data_import import import_data, import_meta
from pf2.figures.common import getSetup
from pf2.predict import predict_mortality
from pf2.tensor import build_tensor, run_parafac2


def makeFigure():
    meta = import_meta()
    data = import_data()
    tensor, patients = build_tensor(data)
    pf2 = run_parafac2(tensor)

    patient_factor = pd.DataFrame(
        pf2.factors[0],
        index=patients.loc[:, "patient_id"],
        columns=np.arange(pf2.rank) + 1,
    )
    patient_factor = patient_factor / patient_factor.max(axis=0)

    meta = meta.loc[~meta.loc[:, "patient_id"].duplicated(), :]
    meta = meta.set_index("patient_id", drop=True)
    patient_factor = patient_factor.loc[
        patient_factor.index.isin(meta.index),
        :
    ]
    labels = patient_factor.index.to_series().replace(
        meta.loc[:, "binary_outcome"]
    )

    probabilities = predict_mortality(patient_factor, labels, proba=True)
    predicted = [0 if prob < 0.5 else 1 for prob in probabilities]
    accuracy = accuracy_score(labels, predicted)

    fpr, tpr, _ = roc_curve(labels, probabilities)
    auc_roc = roc_auc_score(labels, probabilities)

    axs, fig = getSetup(
        (8, 4),
        (1, 1)
    )
    ax = axs[0]

    ax.plot([0, 1], [0, 1], linestyle="--", color="k")
    ax.plot(fpr, tpr)
    ax.text(
        0.99,
        0.01,
        s=f"AUC ROC: {round(auc_roc, 2)}\nAccuracy: {round(accuracy, 2)}",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
    )

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    return fig
