"""Figure 2: R2X Curve"""

import numpy as np
import pandas as pd

from pf2.data_import import convert_to_patients, import_data, import_meta
from pf2.figures.common import getSetup
from pf2.predict import predict_mortality
from pf2.tensor import pf2


def makeFigure():
    meta = import_meta()
    data = import_data()
    data, _ = pf2(data)
    meta = meta.loc[~meta.loc[:, "patient_id"].duplicated(), :]
    meta = meta.set_index("patient_id", drop=True)
    conversions = convert_to_patients(data)

    axs, fig = getSetup((6, 6), (2, 1))

    ranks = np.arange(1, 61)
    r2xs = pd.Series(0, dtype=float, index=ranks)
    accuracies = pd.Series(0, dtype=float, index=ranks)
    labels = None
    for rank in ranks:
        data, r2x = pf2(data, rank, do_embedding=False)
        patient_factor = pd.DataFrame(
            data.uns["Pf2_A"],
            index=conversions,
            columns=np.arange(data.uns["Pf2_A"].shape[1]) + 1,
        )
        patient_factor = patient_factor.loc[patient_factor.index.isin(meta.index), :]
        if labels is None:
            labels = patient_factor.index.to_series().replace(
                meta.loc[:, "binary_outcome"]
            )

        acc, _ = predict_mortality(patient_factor, labels)
        r2xs.loc[rank] = r2x
        accuracies.loc[rank] = acc
        accuracies.to_csv("/home/jchin/BAL-Pf2/output/acc_v_rank.csv")

    # R2X Plots

    axs[0].plot(r2xs.index, r2xs)
    axs[0].grid(True)

    axs[0].set_ylabel("R2X")
    axs[0].set_xlabel("Rank")

    # Accuracy Plots

    axs[1].plot(accuracies.index, accuracies)
    axs[1].grid(True)

    axs[1].set_ylabel("Accuracy")
    axs[1].set_xlabel("Rank")

    return fig
