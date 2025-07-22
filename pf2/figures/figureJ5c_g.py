"""Figure J5c_g: Mortality Ramping Validation"""

import numpy as np
import pandas as pd
from anndata import read_h5ad

from pf2.data_import import convert_to_patients, import_meta
from pf2.figures.common import getSetup
from pf2.data_import import combine_cell_types

CMP_1_GENES = np.array(["NT5E", "TNFSF15", "SCG5", "PADI4"])
CMP_1_CELL_TYPES = np.array(["Macrophages", "Monocytes"])
PATIENTS = [424, 3152, 5469, 6308, 7048]


def makeFigure():
    meta = import_meta(drop_duplicates=False)
    data = read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    sample_conversions = pd.Index(convert_to_patients(data, sample=True).values)

    meta = meta.set_index("sample_id")
    shared_indices = sample_conversions.intersection(meta.index)
    meta = meta.loc[shared_indices, :].sort_values("icu_day", ascending=True)
    data = data[data.obs.loc[:, "patient_id"].isin(PATIENTS), :]
    combine_cell_types(data)

    axs, fig = getSetup((4, len(PATIENTS) * 2), (len(PATIENTS), 1))

    for patient, ax in zip(PATIENTS, axs):
        cell_frac = pd.DataFrame(
            index=np.array(["Component"]),
            columns=meta.loc[meta.loc[:, "patient_id"] == patient, "icu_day"],
            dtype=float,
        )
        for icu_day, sample in zip(
            meta.loc[meta.loc[:, "patient_id"] == patient, "icu_day"],
            meta.loc[meta.loc[:, "patient_id"] == patient, :].index
        ):
            sample_data = data[
                data.obs.loc[:, "sample_id"] == sample,
                CMP_1_GENES
            ]
            n_cells = sample_data.shape[0]
            cell_frac.loc["Component", icu_day] = (
                sample_data[
                    sample_data.X.sum(axis=1) > 0,  # type: ignore
                    :
                ].shape[0]
                / n_cells
            )
            for ct in CMP_1_CELL_TYPES:
                cell_frac.loc[ct, icu_day] = (
                    sample_data[
                        sample_data.obs.loc[:, "combined_cell_type"] == ct
                    ].shape[0]
                    / n_cells
                )

        for index, cell_type in enumerate(cell_frac.index):
            ax.plot(
                cell_frac.columns,
                cell_frac.loc[cell_type, :],
                label=cell_type
            )

        ax.set_yscale("log")
        ax.set_yticks(np.logspace(-3, 0, num=4, base=10))
        ax.set_title(str(patient))
        ax.legend()

    return fig
