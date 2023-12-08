"""Figure 4: Component Association Errorbars"""
import numpy as np
import pandas as pd
from sklearn.utils import resample
from tqdm import tqdm

from pf2.data_import import convert_to_patients, import_data, import_meta
from pf2.figures.common import getSetup
from pf2.predict import predict_mortality
from pf2.tensor import pf2

TRIALS = 30


def makeFigure():
    data = import_data()
    meta = import_meta()
    data, _ = pf2(data, do_embedding=False)
    meta = meta.loc[~meta.loc[:, "patient_id"].duplicated(), :]
    meta = meta.set_index("patient_id", drop=True)

    conversions = convert_to_patients(data)
    patient_factor = pd.DataFrame(
        data.uns["Pf2_A"][0],
        index=conversions,
        columns=np.arange(data.uns["Pf2_rank"]) + 1,
    )
    patient_factor = patient_factor.loc[
        patient_factor.index.isin(meta.index), :
    ]
    labels = patient_factor.index.to_series().replace(
        meta.loc[:, "binary_outcome"]
    )

    coefs = pd.DataFrame(
        index=np.arange(TRIALS) + 1, columns=patient_factor.columns
    )
    for trial in tqdm(range(TRIALS)):
        boot_factors, boot_labels = resample(patient_factor, labels)
        _, coef = predict_mortality(boot_factors, boot_labels)
        coefs.iloc[trial, :] = coef

    axs, fig = getSetup((8, 4), (1, 1))
    ax = axs[0]

    ax.errorbar(
        np.arange(coefs.shape[1]) + 1,
        coefs.mean(axis=0),
        capsize=2,
        yerr=1.96 * coefs.std(axis=0) / np.sqrt(TRIALS),
        linestyle="",
        marker=".",
        zorder=3,
    )
    ax.plot([0, 41], [0, 0], linestyle="--", color="k", zorder=0)

    ax.set_xticks(np.arange(1, 41))
    ax.set_xticklabels(np.arange(1, 41), fontsize=8)

    ax.set_xlim([0, 41])
    ax.grid(True)

    ax.set_ylabel("Logistic Regression Coefficient")
    ax.set_xlabel("PARAFAC2 Component")

    return fig
