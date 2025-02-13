"""Figure J3: Mortality Ramping Validation"""

import numpy as np
import pandas as pd
from anndata import read_h5ad

from pf2.data_import import convert_to_patients, import_meta
from pf2.figures.common import getSetup

CMP_26_GENES = np.array(["FAM111B", "RAD54L", "UHRF1", "MCM10", "EXO1"])
CMP_26_CELL_TYPES = np.array(["CD8 T cells", "Proliferating T cells", "NK/gdT cells"])
PATIENTS = [5429, 5469, 7048, 9441, 8415]


def makeFigure():
    meta = import_meta(drop_duplicates=False)
    data = read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    sample_conversions = pd.Index(convert_to_patients(data, sample=True).values)

    meta = meta.set_index("sample_id")
    shared_indices = sample_conversions.intersection(meta.index)
    meta = meta.loc[shared_indices, :].sort_values("icu_day", ascending=True)
    data = data[data.obs.loc[:, "patient_id"].isin(PATIENTS), CMP_26_GENES]

    axs, fig = getSetup((4, len(PATIENTS) * 2), (len(PATIENTS), 1))

    for patient, ax in zip(PATIENTS, axs):
        cell_frac = pd.DataFrame(
            index=np.array(["Component"]),
            columns=np.arange(sum(meta.loc[:, "patient_id"] == patient)),
            dtype=float,
        )
        for index, sample in enumerate(
            meta.loc[meta.loc[:, "patient_id"] == patient, :].index
        ):
            sample_data = data[data.obs.loc[:, "sample_id"] == sample]
            n_cells = sample_data.shape[0]
            cell_frac.loc["Component", index] = (
                sample_data[
                    sample_data.X.mean(axis=1) >= data.X.mean(), :  # type: ignore
                ].shape[0]
                / n_cells
            )
            for ct in CMP_26_CELL_TYPES:
                cell_frac.loc[ct, index] = (
                    sample_data[sample_data.obs.loc[:, "cell_type"] == ct].shape[0]
                    / n_cells
                )

        for index, cell_type in enumerate(cell_frac.index):
            ax.bar(
                np.arange(
                    index * (cell_frac.shape[1] + 1),
                    index * (cell_frac.shape[1] + 1) + cell_frac.shape[1],
                    1,
                ),
                cell_frac.loc[cell_type, :],
                label=cell_type,
                width=1,
            )
            ax.set_xticks(
                np.arange(
                    cell_frac.shape[1] / 2 - 0.5,
                    cell_frac.shape[0] * (cell_frac.shape[1] + 1) + 1,
                    cell_frac.shape[1] + 1,
                )
            )
            ax.set_xticklabels(
                [f"Sample {i + 1}" for i in np.arange(cell_frac.shape[0])]
            )

        ax.set_title(str(patient))
        ax.legend()

    return fig
