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
    """Get a list of the axis objects and create a figure showing both cell type classifications."""
    # Create a larger figure with 4 rows (2 comparisons × 2 cell type versions)
    ax, f = getSetup((14, 12), (4, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    
    pvalue_results = []
    cell_type_options = ["cell_type", "combined_cell_type"]
    status_comparisons = [
        {"name": "C19", "groups": ["D-C19", "L-C19"], "ref": "D-C19"},
        {"name": "nC19", "groups": ["D-nC19", "L-nC19"], "ref": "D-nC19"}
    ]

    # Function to convert p-value to significance category
    def get_significance(p_val):
        if p_val >= 0.05:
            return "NS"
        elif p_val < 0.001:
            return "***"
        elif p_val < 0.01:
            return "**"
        elif p_val < 0.05:
            return "*"
        else:
            return "NS"

    # Track which axis we're using
    current_ax = 0
    
    for celltype in cell_type_options:
        # For combined cell types, we need to process the data differently
        if celltype == "combined_cell_type":
            X_combined = X.copy()
            combine_cell_types(X_combined)
            df = cell_count_perc_df(X_combined, celltype=celltype, include_control=False)
        else:
            df = cell_count_perc_df(X, celltype=celltype, include_control=False)
        
        for comparison in status_comparisons:
            # Filter for the specific status comparison
            filtered_df = df[df["Status"].isin(comparison["groups"])]
            celltype_order = np.unique(filtered_df["Cell Type"])
            
            # Create plot
            sns.boxplot(
                data=filtered_df,
                x="Cell Type",
                y="Cell Type Percentage",
                hue="Status",
                order=celltype_order,
                showfliers=False,
                ax=ax[current_ax],
            )
            
            # Add appropriate titles/labels
            ax[current_ax].set_title(f"{comparison['name']} - {celltype}")
            rotate_xaxis(ax[current_ax])
            
            # Perform WLS test for each cell type and add significance markers
            for i, cell_type in enumerate(celltype_order):
                cell_type_data = filtered_df[filtered_df["Cell Type"] == cell_type]
                if len(cell_type_data) > 0:
                    pval_df = wls_stats_comparison(
                        cell_type_data, 
                        "Cell Type Percentage", 
                        "Status", 
                        comparison["ref"]
                    )
                    
                    p_val = pval_df["p Value"].iloc[0]
                    significance = get_significance(p_val)
                    
                    pvalue_results.append({
                        'Cell_Type': cell_type,
                        'Classification': celltype,
                        'P_Value': p_val,
                        'Significance': significance,
                        'Comparison': comparison["name"],
                        'Ref_Group': comparison["ref"]
                    })
            
            current_ax += 1
    
    # Print all results in a single dataframe with significance categories
    results_df = pd.DataFrame(pvalue_results)
    print(results_df)
    
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