"""
Figure J4h: TCR Cell Types
"""

import anndata
import datashader as ds
import datashader.transfer_functions as tf
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.utils.sparsefuncs import mean_variance_axis
from statsmodels.stats.multitest import multipletests

from .common import subplotLabel, getSetup
from .commonFuncs.plotGeneral import rotate_xaxis, plot_avegene_cmps
from ..data_import import add_obs, combine_cell_types

COMPONENTS = [10, 14, 16, 19, 23, 25, 34, 35, 41]
GENES = [
    "TRAV10", "TRAV1-1", "TRBV10-3", "TRAV24", "TRBV14", "TRAV30", "TRBV5-5",
    "TRBV13", "TRBV10-2"
]


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    axs, f = getSetup(
        (4 * (len(COMPONENTS) + 1), 4),
        (1, len(COMPONENTS) + 1)
    )

    X = anndata.read_h5ad(
        "/opt/northwest_bal/full_fitted.h5ad"
    )
    combine_cell_types(X)
    X = X[
        X.obs.loc[:, "combined_cell_type"].str.endswith(("T Cells", "T-regulatory")),
        :
    ]
    cell_types = X.obs.loc[:, "cell_type"].unique()

    thresholds = np.percentile(X.obsm["weighted_projections"], 1, axis=0)
    for comp in COMPONENTS:
        X.obs.loc[:, f"Component {comp}"] = False
        index = X.obsm["weighted_projections"][
            :,
            comp - 1
        ] < thresholds[comp - 1]
        X.obs.loc[index, f"Component {comp}"] = True

    for ax, comp in zip(axs[:-1], COMPONENTS):
        comp_cells = X[
            X.obs.loc[:, f"Component {comp}"],
            :
        ].obs.loc[:, "cell_type"].value_counts()
        comp_cells = comp_cells / comp_cells.sum()
        comp_cells = comp_cells.reindex(cell_types)

        other_cells = X[
            ~X.obs.loc[:, f"Component {comp}"],
            :
        ].obs.loc[:, "cell_type"].value_counts()
        other_cells = other_cells / other_cells.sum()
        other_cells = other_cells.reindex(cell_types)

        ax.bar(
            np.arange(0, 3 * len(cell_types), 3),
            comp_cells,
            width=1,
            color="tab:orange",
            label=f"Component {comp}"
        )
        ax.bar(
            np.arange(1, 3 * len(cell_types), 3),
            other_cells,
            width=1,
            color="tab:blue",
            label="Other T Cells"
        )

        ax.set_xticks(np.arange(0.5, 3 * len(cell_types), 3))
        ax.set_xticklabels(
            cell_types,
            rotation=45,
            ha="right",
            ma="right",
            va="top"
        )
        ax.set_title(f"Component {comp}")
        ax.legend()
        ax.set_ylabel("Cell Proportion")

    ax = axs[-1]

    comp_cells = X[
        X.obs.loc[:, [f"Component {comp}" for comp in COMPONENTS]].any(axis=1),
        :
    ].obs.loc[:, "cell_type"].value_counts()
    comp_cells = comp_cells / comp_cells.sum()
    comp_cells = comp_cells.reindex(cell_types)

    other_cells = X[
        ~X.obs.loc[:, [f"Component {comp}" for comp in COMPONENTS]].any(axis=1),
        :
    ].obs.loc[:, "cell_type"].value_counts()
    other_cells = other_cells / other_cells.sum()
    other_cells = other_cells.reindex(cell_types)

    ax.bar(
        np.arange(0, 3 * len(cell_types), 3),
        comp_cells,
        width=1,
        color="tab:orange",
        label="TCR component cells"
    )
    ax.bar(
        np.arange(1, 3 * len(cell_types), 3),
        other_cells,
        width=1,
        color="tab:blue",
        label="Other T cells"
    )

    ax.set_xticks(np.arange(0.5, 3 * len(cell_types), 3))
    ax.set_xticklabels(
        cell_types,
        rotation=45,
        ha="right",
        ma="right",
        va="top"
    )
    ax.set_title("All TCR Components")
    ax.legend()
    ax.set_ylabel("Cell Proportion")

    return f
