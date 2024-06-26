"""Figure 4: Component Association Errorbars"""
import numpy as np
import pandas as pd
from anndata import read_h5ad
from sklearn.utils import resample
from tqdm import tqdm

from pf2.data_import import convert_to_patients, import_meta
from pf2.figures.common import getSetup
from pf2.predict import predict_mortality

TRIALS = 30


def makeFigure():
    meta = import_meta()
    data = read_h5ad("/opt/northwest_bal/full_fitted.h5ad", backed="r")

    meta = meta.loc[~meta.loc[:, "patient_id"].duplicated(), :]
    meta = meta.set_index("patient_id", drop=True)

    conversions = convert_to_patients(data)
    patient_factor = pd.DataFrame(
        data.uns["Pf2_A"],
        index=conversions,
        columns=np.arange(data.uns["Pf2_A"].shape[1]) + 1,
    )
    meta = meta.loc[patient_factor.index, :]

    patient_factor = patient_factor.loc[
        meta.loc[:, "patient_category"] != "Non-Pneumonia Control",
        :
    ]
    meta = meta.loc[
        meta.loc[:, "patient_category"] != "Non-Pneumonia Control",
        :
    ]

    covid_factors = patient_factor.loc[
        meta.loc[:, "patient_category"] == "COVID-19",
        :
    ]
    covid_labels = meta.loc[
        meta.loc[:, "patient_category"] == "COVID-19",
        "binary_outcome"
    ]
    nc_factors = patient_factor.loc[
        meta.loc[:, "patient_category"] != "COVID-19",
        :
    ]
    nc_labels = meta.loc[
        meta.loc[:, "patient_category"] != "COVID-19",
        "binary_outcome"
    ]

    covid_coefficients = pd.DataFrame(
        0,
        dtype=float,
        index=np.arange(TRIALS) + 1,
        columns=patient_factor.columns
    )
    nc_coefficients = covid_coefficients.copy(deep=True)
    for trial in tqdm(range(TRIALS)):
        boot_covid, boot_covid_labels = resample(covid_factors, covid_labels)
        _, covid_coef = predict_mortality(boot_covid, boot_covid_labels)
        boot_nc, boot_nc_labels = resample(nc_factors, nc_labels)
        _, nc_coef = predict_mortality(boot_nc, boot_nc_labels)

        covid_coefficients.loc[trial + 1, covid_coef.index] = covid_coef
        nc_coefficients.loc[trial + 1, nc_coef.index] = nc_coef

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
        label="COVID-19"
    )
    ax.errorbar(
        np.arange(1, nc_coefficients.shape[1] * 3, 3),
        nc_coefficients.mean(axis=0),
        capsize=2,
        yerr=1.96 * nc_coefficients.std(axis=0) / np.sqrt(TRIALS),
        linestyle="",
        marker=".",
        zorder=3,
        label="Non COVID-19"
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
