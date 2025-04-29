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
from ..data_import import add_obs, combine_cell_types, condition_factors_meta
from .commonFuncs.plotGeneral import rotate_xaxis
from ..utilities import cell_count_perc_df


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((10, 10), (2, 2))

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    factors_meta_df = condition_factors_meta(X).reset_index()
    X = add_obs(X, "binary_outcome")
    X = add_obs(X, "patient_category")
    X = X[~X.obs["patient_category"].isin(["Non-Pneumonia Control", "COVID-19"])]
    
    
    combine_cell_types(X)
    
    celltype_count_perc_df = cell_count_perc_df(X, celltype="cell_type")
    
    df = celltype_count_perc_df.loc[celltype_count_perc_df["Cell Type"].isin(["B cells", "pDC"])]
    factors_meta_df = factors_meta_df[factors_meta_df["sample_id"].isin(df["sample_id"])]
    
    print(df)
    cmp = 22
    df = pd.merge(
        df,
        factors_meta_df[["sample_id", f"Cmp. {cmp}"]],
        on="sample_id",
        how="inner"
    )
    
    
    print(df)
    plot_celltype_scatter(merged_df=df, celltype1="B cells", celltype2="pDC", ax=ax[0])
 
    # sns.stripplot(
    #     data=df,
    #     x="Status",
    #     y="Cell Count",
    #     hue="Status",
    #     dodge=True,
    #     ax=ax[0],
    # )
    
    
    # plot_correlation_cmp_cell_count_perc(df, cmp, ax[0], cellPerc=False)


    return f


def plot_correlation_cmp_cell_count_perc(
    merged_df: pd.DataFrame,
    cmp: int,
    ax: Axes,
    cellPerc=True
):
    """Plot Pearson correlation of component weights and cell type percentage, split by Status."""
    # Determine the column to use for cell percentage or count
    cellPercCol = "Cell Type Percentage" if cellPerc else "Cell Count"

    # Prepare the dataframe for correlation analysis
    correlationdf = pd.DataFrame([])
    for status in merged_df["Status"].unique():
        # Calculate Pearson correlation
        pearson_corr, _ = pearsonr(merged_df[f"Cmp. {cmp}"], merged_df[cellPercCol])

        # Append the result to the correlation dataframe
        correlationdf = pd.concat([
            correlationdf,
            pd.DataFrame({
                "Status": [status],
                "Correlation": ["Pearson"],
                "Value": [pearson_corr]
            })
        ])

    # Plot the barplot
    sns.barplot(
        data=correlationdf,
        x="Status",
        y="Value",
        hue="Status",
        ax=ax
    )
    rotate_xaxis(ax)
    ax.set(
        title=f"Pearson Correlation: Cmp. {cmp} vs {cellPercCol}",
        ylabel="Pearson Correlation",
        xlabel="Cell Type"
    )
    
    
def plot_celltype_scatter(
    merged_df: pd.DataFrame,
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
        columns={"Cell Type Percentage": f"{celltype1} Percentage"}
    )
    df2 = merged_df[merged_df["Cell Type"] == celltype2].rename(
        columns={"Cell Type Percentage": f"{celltype2} Percentage"}
    )

    print(df1)
    # Merge the two DataFrames on sample_id and Status
    scatter_df = pd.merge(
        df1[["sample_id", f"{celltype1} Percentage", "Status"]],
        df2[["sample_id", f"{celltype2} Percentage", "Status"]],
        on=["sample_id", "Status"]
    )

    print(scatter_df)
    # Create scatter plot
    sns.scatterplot(
        data=scatter_df,
        x=f"{celltype1} Percentage",
        y=f"{celltype2} Percentage",
        hue="Status",  # Color points by Status
        ax=ax,
        s=100  # Adjust point size
    )

    # Set axis labels and title
    ax.set_xlabel(f"{celltype1} Percentage")
    ax.set_ylabel(f"{celltype2} Percentage")
    ax.set_title(f"Scatter Plot of {celltype1} vs {celltype2} Percentages")

    return scatter_df