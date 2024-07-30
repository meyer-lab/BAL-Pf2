"""Figure 4: Component Association Errorbars"""

import numpy as np
import pandas as pd
from anndata import read_h5ad

from pf2.data_import import convert_to_patients, import_meta
from pf2.figures.common import getSetup
from pf2.predict import predict_mortality

TRIALS = 30


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

    covid_coefficients = pd.DataFrame(
        0, dtype=float, index=np.arange(TRIALS) + 1, columns=patient_factor.columns
    )
    nc_coefficients = covid_coefficients.copy(deep=True)
    for trial in range(TRIALS):
        boot_index = np.random.choice(
            patient_factor.shape[0], replace=True, size=patient_factor.shape[0]
        )
        boot_factor = patient_factor.iloc[boot_index, :]
        boot_meta = meta.iloc[boot_index, :]
        _, (covid_plsr, nc_plsr) = predict_mortality(boot_factor, boot_meta)

        covid_coefficients.loc[trial + 1, covid_plsr.coef_.index] = \
            covid_plsr.coef_
        nc_coefficients.loc[trial + 1, nc_plsr.coef_.index] = \
            nc_plsr.coef_

    axs, fig = getSetup((8, 4), (1, 1))
    ax = axs[0]

    ax.errorbar(
        np.arange(0, covid_coefficients.shape[1] * 3, 3),
        covid_coefficients.mean(axis=0),
        capsize=2,
        yerr=1.96 * covid_coefficients.std(axis=0) / np.sqrt(TRIALS),
        linestyle="",
        marker=".",
        zorder=3,
        label="COVID-19",
    )
    ax.errorbar(
        np.arange(1, nc_coefficients.shape[1] * 3, 3),
        nc_coefficients.mean(axis=0),
        capsize=2,
        yerr=1.96 * nc_coefficients.std(axis=0) / np.sqrt(TRIALS),
        linestyle="",
        marker=".",
        zorder=3,
        label="Non COVID-19",
    )
    ax.plot([-1, 200], [0, 0], linestyle="--", color="k", zorder=0)

    ax.set_xticks(np.arange(0.5, data.uns["Pf2_A"].shape[1] * 3, 3))
    ax.set_xticklabels(np.arange(data.uns["Pf2_A"].shape[1]) + 1, fontsize=8)

    ax.set_xlim([-1, data.uns["Pf2_A"].shape[1] * 3])
    ax.legend()
    ax.grid(True)

    ax.set_ylabel("Logistic Regression Coefficient")
    ax.set_xlabel("PARAFAC2 Component")

    return fig
