"""Figure J4b: TCR Diversity"""
import numpy as np
import pandas as pd
from anndata import read_h5ad

from pf2.data_import import add_obs, condition_factors_meta
from pf2.figures.common import getSetup

GENES = sorted([
    "TRAV10", "TRAV1-1", "TRBV10-3", "TRBV14", "TRBV5-5",
    "TRBV10-2", "TRAV24", "TRAV30", "TRDV1", "TRBV13"
])


def makeFigure():
    data = read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    cfm = condition_factors_meta(data)
    add_obs(data, "binary_outcome")
    add_obs(data, "covid_status")

    data = data[data.obs.loc[:, "covid_status"] == True, :]
    tcr_genes = data.var_names[data.var_names.str.match("TR[ABDG][VC]")]

    outcomes = pd.DataFrame(
        0,
        index=data.obs.loc[:, "sample_id"].unique(),
        columns=GENES + [
            "t_cell_percentage",
            "binary_outcome",
            "patient_id",
            "icu_day"
        ]
    )
    for sample_id in data.obs.loc[:, "sample_id"].unique():
        sample_data = data[data.obs.loc[:, "sample_id"] == sample_id, :]
        data = data[data.obs.loc[:, "sample_id"] != sample_id, :]
        for gene in tcr_genes:
            outcomes.loc[sample_id, gene] = int(
                len(sample_data[:, gene].X.data) / sum(
                    sample_data.obs.loc[:, "cell_type"].str.endswith(
                        "T cells",
                        "Tregs"
                    )
                ) > 0.01
            )

        outcomes.loc[
            sample_id,
            ["binary_outcome", "patient_id"]
        ] = sample_data.obs.loc[
            :,
            ["binary_outcome", "patient_id"]
        ].iloc[0, :]
        outcomes.loc[sample_id, "t_cell_percentage"] = sum(sample_data.obs.loc[
            :, "cell_type"
        ].str.endswith(("T cells", "Tregs"))) / sample_data.shape[0]
        outcomes.loc[sample_id, "icu_day"] = cfm.loc[sample_id, "ICU Day"]

    axs, fig = getSetup(
        (16, 4),
        (1, 4)
    )
    ax = axs[0]

    survived = outcomes.loc[outcomes.loc[:, "binary_outcome"] == 0, :]
    deceased = outcomes.loc[outcomes.loc[:, "binary_outcome"] == 1, :]

    for index, gene in enumerate(GENES):
        ax.errorbar(
            4 * index,
            survived.loc[:, gene].mean(),
            marker="o",
            markersize=5,
            yerr=1.96 * survived.loc[:, gene].std() / np.sqrt(
                survived.loc[:, gene].shape[0]
            ),
            capsize=3,
            color="tab:green"
        )
        ax.errorbar(
            4 * index + 1,
            deceased.loc[:, gene].mean(),
            marker="o",
            markersize=5,
            yerr=1.96 * deceased.loc[:, gene].std() / np.sqrt(
                deceased.loc[:, gene].shape[0]
            ),
            capsize=3,
            color="tab:red"
        )

    ax.legend(["Survived", "Deceased"])
    ax.set_xticks(np.arange(0.5, 4 * len(GENES), 4))
    ax.set_xticklabels(GENES)
    ax.set_ylabel("Percentage Expressing")

    ax = axs[1]

    ax.errorbar(
        0,
        survived.loc[:, "t_cell_percentage"].mean(),
        marker="o",
        markersize=5,
        yerr=1.96 * survived.loc[:, "t_cell_percentage"].std() / np.sqrt(
            survived.loc[:, "t_cell_percentage"].shape[0]
        ),
        capsize=3,
        color="tab:green"
    )
    ax.errorbar(
        1,
        deceased.loc[:, "t_cell_percentage"].mean(),
        marker="o",
        markersize=5,
        yerr=1.96 * deceased.loc[:, "t_cell_percentage"].std() / np.sqrt(
            deceased.loc[:, "t_cell_percentage"].shape[0]
        ),
        capsize=3,
        color="tab:red"
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Survived", "Deceased"])
    ax.set_ylabel("T Cell Percentage")
    ax.set_xlim([-0.5, 1.5])

    ax = axs[2]

    _survived = survived.loc[
        :,
        GENES
    ]
    _deceased = deceased.loc[
        :,
        GENES
    ]
    ax.errorbar(
        0,
        (_survived.sum(axis=1) / _survived.shape[1]).mean(),
        marker="o",
        markersize=5,
        yerr=1.96 * (
            _survived.sum(axis=1) / _survived.shape[1]).std() / np.sqrt(
            _survived.shape[0]
        ),
        capsize=3,
        color="tab:green"
    )
    ax.errorbar(
        1,
        (_deceased.sum(axis=1) / _deceased.shape[1]).mean(),
        marker="o",
        markersize=5,
        yerr=1.96 * (
            _deceased.sum(axis=1) / _deceased.shape[1]).std() / np.sqrt(
            _deceased.shape[0]
        ),
        capsize=3,
        color="tab:red"
    )

    for index, region in enumerate(
        tcr_genes.str[3].unique(),
        start=1
    ):
        _survived = survived.loc[
            :,
            tcr_genes[tcr_genes.str[3] == region]
        ]
        _deceased = deceased.loc[
            :,
            tcr_genes[tcr_genes.str[3] == region]
        ]
        ax.errorbar(
            4 * index,
            (_survived.sum(axis=1) / _survived.shape[1]).mean(),
            marker="o",
            markersize=5,
            yerr=1.96 * (_survived.sum(axis=1) / _survived.shape[1]).std() / np.sqrt(
                _survived.shape[0]
            ),
            capsize=3,
            color="tab:green"
        )
        ax.errorbar(
            4 * index + 1,
            (_deceased.sum(axis=1) / _deceased.shape[1]).mean(),
            marker="o",
            markersize=5,
            yerr=1.96 * (_deceased.sum(axis=1) / _deceased.shape[1]).std() / np.sqrt(
                _deceased.shape[0]
            ),
            capsize=3,
            color="tab:red"
        )

    for index, receptor in enumerate(
        tcr_genes.str[2].unique(),
        start=len(tcr_genes.str[3].unique()) + 1
    ):
        _survived = survived.loc[
            :,
            tcr_genes[tcr_genes.str[2] == receptor]
        ]
        _deceased = deceased.loc[
            :,
            tcr_genes[tcr_genes.str[2] == receptor]
        ]
        ax.errorbar(
            4 * index,
            (_survived.sum(axis=1) / _survived.shape[1]).mean(),
            marker="o",
            markersize=5,
            yerr=1.96 * (_survived.sum(axis=1) / _survived.shape[1]).std() / np.sqrt(
                _survived.shape[0]
            ),
            capsize=3,
            color="tab:green"
        )
        ax.errorbar(
            4 * index + 1,
            (_deceased.sum(axis=1) / _deceased.shape[1]).mean(),
            marker="o",
            markersize=5,
            yerr=1.96 * (_deceased.sum(axis=1) / _deceased.shape[1]).std() / np.sqrt(
                _deceased.shape[0]
            ),
            capsize=3,
            color="tab:red"
        )

    tick_labels = ["Cmp Genes"] + list(
        "Region: " + tcr_genes.str[3].unique()
    ) + list(
        "Receptor: " + tcr_genes.str[2].unique()
    )

    ax.set_xticks(np.arange(0.5, 4 * len(tick_labels), 4))
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Percent Expressed")

    ax = axs[3]

    sample_counts = outcomes.loc[:, "patient_id"].value_counts()
    patients = sample_counts.loc[sample_counts >= 3].index
    outcomes = outcomes.loc[outcomes.loc[:, "patient_id"].isin(patients), :]
    outcomes = outcomes.sort_values(
        by=["patient_id", "icu_day"],
        ascending=True
    )
    tcr_diversity = pd.DataFrame(
        index=outcomes.loc[:, "patient_id"].unique(),
        columns=list(np.arange(1, 4)),
        dtype=int
    )
    for patient in outcomes.loc[:, "patient_id"].unique():
        patient_data = outcomes.loc[outcomes.loc[:, "patient_id"] == patient, :]
        tcr_diversity.loc[patient, :] = patient_data.loc[
            :,
            GENES
        ].iloc[
            -3:,
            :
        ].sum(axis=1).values

    tcr_diversity.loc[:, "binary_outcome"] = outcomes.loc[
        ~outcomes.loc[:, "patient_id"].duplicated(),
        "binary_outcome"
    ].values

    ax.errorbar(
        np.arange(-0.1, 2),
        tcr_diversity.loc[
            tcr_diversity.loc[:, "binary_outcome"] == 0,
            :3
        ].mean(axis=0),
        yerr=1.96 * tcr_diversity.loc[
            tcr_diversity.loc[:, "binary_outcome"] == 0,
            :3
        ].std(axis=0) / np.sqrt(
            sum(tcr_diversity.loc[:, "binary_outcome"] == 0)
        ),
        color="tab:green",
        label="Survived",
        marker="o",
        markersize=5,
        capsize=3
    )
    ax.errorbar(
        np.arange(0.1, 3),
        tcr_diversity.loc[
            tcr_diversity.loc[:, "binary_outcome"] == 1,
            :3
        ].mean(axis=0),
        yerr=1.96 * tcr_diversity.loc[
            tcr_diversity.loc[:, "binary_outcome"] == 1,
            :3
        ].std(axis=0) / np.sqrt(
            sum(tcr_diversity.loc[:, "binary_outcome"] == 1)
        ),
        color="tab:red",
        label="Deceased",
        marker="o",
        markersize=5,
        capsize=3
    )
    ax.legend()

    return fig
