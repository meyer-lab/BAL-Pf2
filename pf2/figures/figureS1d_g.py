"""Figure S1: Cell type abundance and distribution across patient statuses"""

import anndata
import numpy as np
import seaborn as sns
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotGeneral import rotate_xaxis
from ..data_import import add_obs, combine_cell_types
from ..utilities import cell_count_perc_df, perform_statistical_tests
import pandas as pd


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((14, 6), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    combine_cell_types(X)

    celltype = ["cell_type", "combined_cell_type"]
    status_comparisons = [
        {"name": "C19", "groups": ["D-C19", "L-C19"], "ref": "D-C19"},
        {"name": "nC19", "groups": ["D-nC19", "L-nC19"], "ref": "D-nC19"}
    ]
    pvalue_results = []
    axs = 0
    
    for i, celltypes in enumerate(celltype):
        celltype_count_perc_df = cell_count_perc_df(X, celltype=celltypes, include_control=False)
        celltype = np.unique(celltype_count_perc_df["Cell Type"])
        for comparison in status_comparisons:
            filtered_df = celltype_count_perc_df[celltype_count_perc_df["Status"].isin(comparison["groups"])]
            celltype_order = np.unique(filtered_df["Cell Type"])
        
            sns.boxplot(
                data=filtered_df,
                x="Cell Type",
                y="Cell Type Percentage",
                hue="Status",
                order=celltype_order,
                showfliers=False,
                ax=ax[axs],
            )
            rotate_xaxis(ax[axs])

            comparison_results = perform_statistical_tests(filtered_df, celltypes, status_comparisons)
            pvalue_results.append(comparison_results)
            axs += 1

    results_df = pd.concat(pvalue_results)
    print(results_df)

    return f
