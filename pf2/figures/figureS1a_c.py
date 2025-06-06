"""Figure S1: Cell type abundance and distribution across patient statuses"""

import anndata
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotGeneral import rotate_xaxis, add_stat_annotation
from ..data_import import add_obs, combine_cell_types
from ..utilities import bal_combine_bo_covid, cell_count_perc_df, wls_stats_comparison
import pandas as pd

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((14, 6), (3, 3))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    combine_cell_types(X)

    # plot_cell_count(X, ax[0])
    pvalue_results = []

    # celltype = ["cell_type", "combined_cell_type"]
    celltype = ["combined_cell_type"]
    status_filters = [["D-C19", "L-C19"], ["D-nC19", "L-nC19"]]  # C19 categories and nC19 categories

    axs = 0
    for i, celltypes in enumerate(celltype):
        celltype_count_perc_df = cell_count_perc_df(X, celltype=celltypes, include_control=False)
        for j, filt in enumerate(status_filters[i]):
            # Filter for the specific status categories for this plot
            filtered_df = celltype_count_perc_df[
                celltype_count_perc_df["Status"].isin(status_filters[i])
            ]

            print(filtered_df)
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
            
            # Set reference point for statistical comparison
            reference_group = "D-C19" if j == 0 else "D-nC19"  # First plot (C19) vs D-nC19, Second plot (nC19) vs D-C19
            
            # Perform WLS test for each cell type
            for k, cell_type in enumerate(celltype_order):
                # print(filtered_df)
                cell_type_data = filtered_df[filtered_df["Cell Type"] == cell_type]
                print(cell_type)
                print(cell_type_data)
                print(f"Reference Group: {reference_group}")
                if len(cell_type_data) > 0:
                    pval_df = wls_stats_comparison(
                        cell_type_data, 
                        "Cell Type Percentage", 
                        "Status", 
                        reference_group
                    )
                    # print(pval_df)
    
                    pvalue_results.append({
                        'Cell_Type': cell_type,
                        'P_Value': pval_df["p Value"].iloc[0],
                        'Cell_Type_Category': celltypes,
                        'Ref_Group': reference_group
                    })
            axs += 1
            
            
        # ax[3].remove()
        
    print(pd.DataFrame(pvalue_results))
    # print("P-Value Results:")
    # for result in pvalue_results:
        # print(f"Cell Type: {result['Cell_Type']}, P-Value: {result['P_Value']}, Category: {result['Cell_Type_Category']}, Reference Group: {result['Ref_Group']}")
    
    return f


def plot_cell_count(
    X: anndata.AnnData,
    ax: Axes,
    cond: str = "sample_id",
    status1: str = "binary_outcome",
    status2: str = "patient_category",
):
    """Plots overall cell count."""
    df = X.obs[[cond, status1, status2]].reset_index(drop=True)

    df = bal_combine_bo_covid(df)
    dfCond = (
        df.groupby([cond, "Status"], observed=True)
        .size()
        .reset_index(name="Cell Count")
    )

    sns.barplot(data=dfCond, x="Status", y="Cell Count", hue="Status", ax=ax)
    rotate_xaxis(ax)