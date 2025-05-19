"""Table S1: Summary statistics for metadata correlates."""

import anndata
from .common import getSetup
from ..data_import import meta_raw_df
import pandas as pd
import seaborn as sns
from ..utilities import bal_combine_bo_covid, cell_count_perc_df
from ..data_import import add_obs, combine_cell_types
from .commonFuncs.plotGeneral import rotate_xaxis
from matplotlib.axes import Axes
import scipy.stats as stats
import statsmodels.api as sm
import numpy as np

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((8, 8), (3, 3))

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    combine_cell_types(X)

    all_meta_df = meta_raw_df(X, all=True)
    all_meta_df = bal_combine_bo_covid(all_meta_df)
    categories_to_exclude = ["Non-Pneumonia Control"]
    all_meta_df = all_meta_df[~all_meta_df["patient_category"].isin(categories_to_exclude)]
    meta = "BAL_pct_neutrophils"
    corr_df = all_meta_df[["Status", meta]].dropna()
    
    corr_df = corr_df[corr_df["Status"].isin(["D-C19", "L-C19"])] 
    sns.violinplot(corr_df, x="Status", y=meta, ax=ax[0], hue="Status")
    
    group_dc19 = corr_df[corr_df["Status"] == "D-C19"][meta]
    group_lc19 = corr_df[corr_df["Status"] == "L-C19"][meta]
    t_stat, p_value = stats.ttest_ind(group_dc19, group_lc19)
    
    if p_value < 0.001:
        p_text = "*** p < 0.001"
    elif p_value < 0.01:
        p_text = "** p < 0.01"
    elif p_value < 0.05:
        p_text = "* p < 0.05"
    else:
        p_text = f"p = {p_value:.3f}"

    # Add annotation above the plot
    ax[0].text(0.5, 0.95, f"t-test: {p_text}", 
            ha='center', va='center', 
            bbox=dict(facecolor='white', alpha=0.8))

    X = X[~X.obs["patient_category"].isin(["Non-Pneumonia Control", "Other Viral Pneumonia", "Other Pneumonia"])]
    celltype = ["cell_type", "cell_type"]
    types = ["Cell Count", "Cell Type Percentage"]
    
    axs=1
    for i, celltypes in enumerate(celltype):
        for j, type in enumerate(types):
            celltype_count_perc_df = cell_count_perc_df(X, celltype=celltypes, include_control=False)
            if i == 0:
                t_cells = ["CD4 T cells", "CD8 T cells","CM CD8 T cells", "IFN respons. CD8 T cells", "Tregs"]  
                
            if i == 1:
                t_cells = ["CD4 T cells", "CD8 T cells","CM CD8 T cells", "IFN respons. CD8 T cells", "Tregs"] 
                t_cell_mapping = {
                "CD4 T cells": "CD4 T cells",  # Keep as is
                "CD8 T cells": "CD8 T cells",   # Keep as is
                "CM CD8 T cells": "CD8 T cells",  # Merge into CD8
                "IFN respons. CD8 T cells": "CD8 T cells",  # Merge into CD8
                "Tregs": "CD4 T cells"  # Keep as is
            }
                
            df = celltype_count_perc_df[celltype_count_perc_df["Cell Type"].isin(t_cells)].copy().reset_index(drop=True)
        
            if i == 1:
                print(df)
                df["Cell Type"] = df["Cell Type"].map(t_cell_mapping)
                print(df)
                
                
            df["Cell Type"] = df["Cell Type"].astype(str)
            
            print(df)
            final_df = df.reset_index(drop=True)
            
       
                
            
            sns.boxplot(
                data=final_df,
                x="Cell Type",
                y=type,
                hue="Status",
                # order=celltype,
                showfliers=False,
                ax=ax[axs],
            )
            rotate_xaxis(ax[axs])
            
            if type == "Cell Type Percentage":
                pval_df = wls_stats_comparison(final_df, type, "Status", "D-C19")
                add_pvalue_annotation(ax[axs], pval_df)
                
                
            if type == "Cell Count":
                # Get unique cell types
                unique_cell_types = final_df["Cell Type"].unique()
                
                # Perform t-test for each cell type between Status groups
                for i, cell_type in enumerate(unique_cell_types):
                    # Extract data for this cell type
                    group1 = final_df[(final_df["Cell Type"] == cell_type) & 
                                    (final_df["Status"] == final_df["Status"].unique()[0])][type]
                    group2 = final_df[(final_df["Cell Type"] == cell_type) & 
                                    (final_df["Status"] == final_df["Status"].unique()[1])][type]
                    
                    # Perform t-test
                    t_stat, p_val = stats.ttest_ind(group1, group2)
                    
                    
                    # Determine significance stars
                    y_max = ax[axs].get_ylim()[1]

                    # Determine where to place the p-value bars
                    spacing = y_max * 0.1
                    if p_val < 0.001:
                        stars = '***'
                    elif p_val < 0.01:
                        stars = '**'
                    elif p_val < 0.05:
                        stars = '*'
                    else:
                        stars = 'ns'

                    # Position for annotation
                    y_position = y_max + spacing * (i + 1)

                    # Add p-value text
                    ax[axs].text(i, y_position, f'p = {p_val:.3f} {stars}', 
                            horizontalalignment='center', 
                            verticalalignment='bottom')

                    # Optional: Add a horizontal line to indicate p-value comparison
                    ax[axs].plot([i-0.3, i+0.3], [y_position]*2, color='black', linewidth=1)
                            
            
            
            axs+=1
            
        


    return f


def wls_stats_comparison(df, column_comparison_name, category_name, status_name):
    """Calculates whether cells are statistically signicantly different"""
    pval_df = pd.DataFrame()
    df["Y"] = 1
    df.loc[df[category_name] == status_name, "Y"] = 0
    for cell in df["Cell Type"].unique():
        Y = df.loc[df["Cell Type"] == cell][column_comparison_name].to_numpy()
        X = df.loc[df["Cell Type"] == cell]["Y"].to_numpy()
        weights = np.power(df.loc[df["Cell Type"] == cell]["Cell Count"].values, 1)
        mod_wls = sm.WLS(Y, sm.tools.tools.add_constant(X), weights=weights)
        res_wls = mod_wls.fit()
        print(res_wls.pvalues)
        pval_df = pd.concat(
            [
                pval_df,
                pd.DataFrame(
                    {
                        "Cell Type": [cell],
                        "p Value": res_wls.pvalues[1]
                        * df["Cell Type"].unique().size,
                    }
                ),
            ]
        )

    return pval_df


def add_pvalue_annotation(ax, pval_df):
    # Get the current y-axis limit
    y_max = ax.get_ylim()[1]

    # Determine where to place the p-value bars
    spacing = y_max * 0.1

    # Iterate through unique cell types
    for i, cell_type in enumerate(pval_df.index):
        p_value = pval_df.loc[cell_type, 'p Value']


        # Determine significance stars
        if p_value.iloc[i] < 0.001:
            stars = '***'
        elif p_value.iloc[i] < 0.01:
            stars = '**'
        elif p_value.iloc[i] < 0.05:
            stars = '*'
        else:
            stars = 'ns'

        # Position for annotation
        y_position = y_max + spacing * (i + 1)

        # Add p-value text
        ax.text(i, y_position, f'p = {p_value.iloc[i]:.3f} {stars}', 
                horizontalalignment='center', 
                verticalalignment='bottom')

        # Optional: Add a horizontal line to indicate p-value comparison
        ax.plot([i-0.3, i+0.3], [y_position]*2, color='black', linewidth=1)