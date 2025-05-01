"""
Figure S5:
"""


import anndata
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr
from matplotlib.axes import Axes
from .common import getSetup
from ..data_import import add_obs, combine_cell_types, condition_factors_meta, condition_factors_meta_raw
from .commonFuncs.plotGeneral import rotate_xaxis
from ..utilities import cell_count_perc_df


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # ax, f = getSetup((20, 20), (2, 2))
    ax, f = getSetup((8, 8), (2, 2))

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    factors_meta_df, _ = condition_factors_meta_raw(X)
    factors_meta_df = factors_meta_df.reset_index()
    X = add_obs(X, "binary_outcome")
    X = add_obs(X, "patient_category")
    X = X[~X.obs["patient_category"].isin(["Non-Pneumonia Control", "COVID-19"])]
    
    combine_cell_types(X)
    celltype_count_perc_df = cell_count_perc_df(X, celltype="cell_type")
    
    #### Plot scatter plot of B cells vs pDC percentages
 
    df = celltype_count_perc_df.loc[celltype_count_perc_df["Cell Type"].isin(["B cells", "pDC"])]
    axs = 0
    for i, column in enumerate(["Cell Count", "Cell Type Percentage"]):
        merged_df = pd.merge(
            df,
            factors_meta_df[["sample_id", "Cmp. 22", "Cmp. 62", "icu_day", "immunocompromised_flag", "episode_etiology"]],
            on="sample_id",
            how="inner"
        )
        # Merge icu days into categroy 
        merged_df["icu_day"] = pd.cut(
            merged_df["icu_day"],
            bins=[1, 7, 27, 100],
            labels=["1-7", "8-27", "27+"]
        )
        # merged_df = merged_df[merged_df["ICU Day"] > 5]
        sns.scatterplot(merged_df, x="Cmp. 22", y="Cmp. 62", hue="Status", style="icu_day", ax=ax[axs])
        ax[axs].set_title(f"pearson: {pearsonr(merged_df["Cmp. 22"], merged_df["Cmp. 62"])[0]:.2f}")
        
        # print(merged_df)
        # a
        # plot_celltype_scatter(merged_df=merged_df, columns=column, celltype1="B cells", celltype2="pDC", ax=ax[axs])
        axs += 1

    
    #### Plot stripplot of cell counts pDC/ B cells
    # axs=0
    # for i, celltype in enumerate(["B cells", "pDC"]):
    #     for j, type in enumerate(["Cell Count", "Cell Type Percentage"]):
    #         df = celltype_count_perc_df.loc[celltype_count_perc_df["Cell Type"] == celltype]
    #         merged_df = pd.merge(
    #             df,
    #             factors_meta_df[["sample_id"]],
    #             on="sample_id",
    #             how="inner"
    #         )
    #         sns.stripplot(
    #             data=merged_df,
    #             x="Status",
    #             y=type,
    #             hue="Status",
    #             dodge=True,
    #             ax=ax[axs],
    #         )
    #         ax[axs].set_title(f"{celltype} {type}")
    #         axs += 1
    
    #### Plot correlation of component weights and cell type percentage for pDC/B cells
    # axs=0
    # for i, celltype in enumerate(["B cells", "pDC"]):
    #     for j, type in enumerate(["Cell Count", "Cell Type Percentage"]):
    #         df = celltype_count_perc_df.loc[celltype_count_perc_df["Cell Type"] == celltype]
    #         merged_df = pd.merge(
    #             df,
    #             factors_meta_df[["sample_id"] + [f"Cmp. {i+1}" for i in range(80)]],
    #             on="sample_id",
    #             how="inner"
    #         )
    #         plot_correlation_all_cmps(merged_df=merged_df, ax=ax[axs], cellPerc=(type == "Cell Type Percentage"))
    #         axs += 1



    return f


def plot_correlation_all_cmps(
    merged_df: pd.DataFrame,
    ax: Axes,
    cellPerc=True
):
    """Plot Pearson correlation of all component weights and cell type percentage, split by Status."""
    # Determine the column to use for cell percentage or count
    cellPercCol = "Cell Type Percentage" if cellPerc else "Cell Count"

    # Prepare the dataframe for correlation analysis
    correlationdf = pd.DataFrame([])
    for cmp in range(1, 81):  # Iterate over all components (Cmp. 1 to Cmp. 80)
        cmp_col = f"Cmp. {cmp}"

        for status in merged_df["Status"].unique():
            # Filter data for the specific status
            status_df = merged_df[merged_df["Status"] == status]

            pearson_corr, _ = pearsonr(status_df[cmp_col], status_df[cellPercCol])
            correlationdf = pd.concat([
                correlationdf,
                pd.DataFrame({
                    "Component": [cmp],
                    "Status": [status],
                    "Correlation": [pearson_corr]
                })
            ])

    sns.barplot(
        data=correlationdf,
        x="Component",
        y="Correlation",
        hue="Status",
        ax=ax
    )
    ax.set_ylim(-1, 1)
    rotate_xaxis(ax)
    ax.set(
        title=f"Pearson Correlation: All Components vs {cellPercCol}",
        ylabel="Pearson Correlation",
        xlabel="Component"
    )
    
    
def plot_celltype_scatter(
    merged_df: pd.DataFrame,
    columns: str, 
    celltype1: str,
    celltype2: str,
    ax: Axes,
):
    """Plots a scatter plot of cell percentages for two cell types, labeled by Status.

    Args:
        merged_df (pd.DataFrame): DataFrame containing cell type percentages and metadata.
        celltype1 (str): Name of the first cell type (x-axis).
        celltype2 (str): Name of the second cell type (y-axis).
        ax (matplotlib.axes.Axes): Axis to plot on.
    """
    # Filter the DataFrame for the two cell types
    df1 = merged_df[merged_df["Cell Type"] == celltype1].rename(
        columns={columns: f"{celltype1} {columns}"}
    )
    df2 = merged_df[merged_df["Cell Type"] == celltype2].rename(
        columns={columns: f"{celltype2} {columns}"}
    )

    # Merge the two DataFrames on sample_id and Status
    scatter_df = pd.merge(
        df1[["sample_id", f"{celltype1} {columns}", "Status"]],
        df2[["sample_id", f"{celltype2} {columns}", "Status"]],
        on=["sample_id", "Status"]
    )
    

    # Create scatter plot
    sns.scatterplot(
        data=scatter_df,
        x=f"{celltype1} {columns}",
        y=f"{celltype2} {columns}",
        hue="Status",  # Color points by Status
        ax=ax,
        s=100  # Adjust point size
    )

    # Set axis labels and title
    ax.set_xlabel(f"{celltype1} {columns}")
    ax.set_ylabel(f"{celltype2} {columns}")
    ax.set_title(f"Scatter Plot of {celltype1} vs {celltype2} {columns}: Pearson: {pearsonr(scatter_df[f'{celltype1} {columns}'], scatter_df[f'{celltype2} {columns}'])[0]:.2f}")

    return scatter_df