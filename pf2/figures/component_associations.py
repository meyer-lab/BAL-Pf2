from os.path import abspath, dirname, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import resample
from tqdm import tqdm

from pf2.data_import import import_data, import_meta
from pf2.predict import predict_mortality
from pf2.tensor import build_tensor, run_parafac2

TRIALS = 30
REPO_PATH = dirname(dirname(abspath(__file__)))


def main():
    meta = import_meta()
    adata = import_data()
    tensor, patients = build_tensor(adata)
    pf2 = run_parafac2(tensor)

    patient_factor = pd.DataFrame(
        pf2.factors[0],
        index=patients.loc[:, "patient_id"],
        columns=np.arange(pf2.rank) + 1,
    )
    patient_factor = patient_factor / patient_factor.max(axis=0)

    meta = meta.loc[~meta.loc[:, "patient_id"].duplicated(), :]
    meta = meta.set_index("patient_id", drop=True)
    patient_factor = patient_factor.loc[patient_factor.index.isin(meta.index), :]
    labels = patient_factor.index.to_series().replace(meta.loc[:, "binary_outcome"])

    coefs = pd.DataFrame(index=np.arange(TRIALS) + 1, columns=patient_factor.columns)
    for trial in tqdm(range(TRIALS)):
        boot_factors, boot_labels = resample(patient_factor, labels)
        _, coef = predict_mortality(boot_factors, boot_labels)
        coefs.iloc[trial, :] = coef

    fig, ax = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True, dpi=300)

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

    plt.savefig(join(REPO_PATH, "output", "component_associations.png"))
    plt.show()


if __name__ == "__main__":
    main()