"""Figure J7b: Protective TCR Subsets"""
from os.path import join

import numpy as np
import seaborn as sns
from anndata import read_h5ad
from scipy.stats import ttest_ind

from pf2.data_import import condition_factors_meta, import_meta
from pf2.figures.common import getSetup

DATA_PATH = join("/opt", "northwest_bal")

COMPONENTS = np.array([10, 14, 16, 23, 34, 41, 19, 25, 12, 35])


def makeFigure():
    data = read_h5ad("/opt/northwest_bal/full_fitted.h5ad", backed="r")
    cond_fact_meta_df = condition_factors_meta(data)
    cond_fact_meta_df = cond_fact_meta_df.iloc[
        :,
        COMPONENTS - 1
    ]
    avg_tcr = cond_fact_meta_df.mean(axis=1).to_frame(name="tcr_avg")

    meta = import_meta(sample_index=True)
    meta = meta.loc[cond_fact_meta_df.index, :]
    meta.loc[:, "covid_status"] = meta.loc[
        :,
        "covid_status"
    ].fillna(False)
    meta = meta.loc[
        meta.loc[:, "covid_status"],
        :
    ]

    meta.loc[:, "secondary_bacterial"] = False
    to_drop = []
    for patient_id in meta.loc[:, "patient_id"].unique():
        if (meta.loc[
            meta.loc[:, "patient_id"] == patient_id,
            "episode_etiology"
        ] == "Bacterial/viral").any():
            meta.loc[
                meta.loc[:, "patient_id"] == patient_id,
                "secondary_bacterial"
            ] = True
        elif (meta.loc[
            meta.loc[:, "patient_id"] == patient_id,
            "episode_etiology"
        ].isna()).all():
            to_drop.append(patient_id)

    meta = meta.loc[~meta.loc[:, "patient_id"].isin(to_drop), :]

    avg_tcr = avg_tcr.loc[meta.index, :]
    avg_tcr.loc[:, "secondary_bacterial"] = meta.loc[
        :,
        "secondary_bacterial"
    ]
    avg_tcr.loc[:, "binary_outcome"] = meta.loc[
        :,
        "binary_outcome"
    ]

    axs, fig = getSetup(
        (12, 4),
        (1, 3)
    )

    # Bacterial/viral

    ax = axs[0]

    bv_cases = avg_tcr.loc[
        avg_tcr.loc[:, "secondary_bacterial"],
        :
    ]
    sns.violinplot(
        bv_cases,
        x="binary_outcome",
        y="tcr_avg",
        ax=ax
    )
    bv_result = ttest_ind(
        bv_cases.loc[bv_cases.loc[:, "binary_outcome"] == 0, "tcr_avg"],
        bv_cases.loc[bv_cases.loc[:, "binary_outcome"] == 1, "tcr_avg"]
    )
    ax.set_title(f"Bacterial/Viral (p-value: {bv_result.pvalue})")

    ax.set_xticklabels(["Survived", "Deceased"])
    ax.set_ylabel("Avg. Component Association")

    # Viral only

    ax = axs[1]

    v_cases = avg_tcr.loc[
        ~avg_tcr.loc[:, "secondary_bacterial"],
        :
    ]
    sns.violinplot(
        v_cases,
        x="binary_outcome",
        y="tcr_avg",
        ax=ax
    )
    v_result = ttest_ind(
        v_cases.loc[v_cases.loc[:, "binary_outcome"] == 0, "tcr_avg"],
        v_cases.loc[v_cases.loc[:, "binary_outcome"] == 1, "tcr_avg"]
    )
    ax.set_title(f"Viral (p-value: {v_result.pvalue})")

    ax.set_xticklabels(["Survived", "Deceased"])
    ax.set_ylabel("Avg. Component Association")

    # Case counts

    ax = axs[2]
    ax.bar(
        np.arange(2),
        [bv_cases.shape[0], v_cases.shape[0]]
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Mixed Viral/Bacterial", "Viral Only"])
    ax.set_ylabel("Cases")

    return fig
