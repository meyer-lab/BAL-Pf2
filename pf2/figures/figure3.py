"""Figure 3: ROC Curves"""

import numpy as np
import pandas as pd
from anndata import read_h5ad
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from pf2.data_import import convert_to_patients, import_meta
from pf2.figures.common import getSetup
from pf2.predict import predict_mortality


def makeFigure():
    meta = import_meta()
    data = read_h5ad("/opt/northwest_bal/full_fitted.h5ad", backed="r")
    conversions = convert_to_patients(data)

    patient_factor = pd.DataFrame(
        data.uns["Pf2_A"],
        index=conversions,
        columns=np.arange(data.uns["Pf2_A"].shape[1]) + 1,
    )
    meta = meta.loc[patient_factor.index, :]

    axs, fig = getSetup((4, 4), (1, 1))
    ax = axs[0]

    probabilities, labels = predict_mortality(patient_factor, meta, proba=True)

    predicted = [0 if prob < 0.5 else 1 for prob in probabilities]
    accuracy = accuracy_score(labels, predicted)

    fpr, tpr, _ = roc_curve(labels, probabilities)
    auc_roc = roc_auc_score(labels, probabilities)

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
