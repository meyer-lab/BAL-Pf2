"""Figure S9"""

import anndata
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotGeneral import plot_correlation_heatmap
from ..data_import import meta_raw_df, add_obs, combine_cell_types, find_overlap_meta_cc
from ..correlation import correlation_meta_cc_df
from ..utilities import cell_count_perc_df


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((15, 6), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    combine_cell_types(X)
    
    cell_comp_df = cell_count_perc_df(X, celltype="combined_cell_type")
    cell_comp_df = cell_comp_df.pivot(
        index=["sample_id"],
        columns="Cell Type",
        values="Cell Type Percentage",
    )
    cell_comp_df = cell_comp_df.fillna(0)
    
    all_meta_df = meta_raw_df(X, all=True)
    
    cell_comp_df, cell_comp_c19_df, cell_comp_nc19_df = find_overlap_meta_cc(cell_comp_df, all_meta_df)
        
    c19_meta_df, nc19_meta_df = meta_raw_df(X, all=False)
    meta = [all_meta_df, c19_meta_df, nc19_meta_df]
    cell_comp = [cell_comp_df.drop(columns=["patient_category"]), cell_comp_c19_df, cell_comp_nc19_df]
    
    for i in range(3):
        corr_df = correlation_meta_cc_df(cell_comp[i], meta[i])
        plot_correlation_heatmap(corr_df, xticks=corr_df.columns, 
                                yticks=corr_df.index, ax=ax[i])
        
    labels = ["All", "C19", "nC19"]
    for i in range(3):
        ax[i].set(title=labels[i])

    return f

