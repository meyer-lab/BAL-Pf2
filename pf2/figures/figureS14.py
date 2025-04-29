"""Figure 14"""

import anndata
import numpy as np
import seaborn as sns
import pandas as pd
from .common import subplotLabel, getSetup
from ..data_import import add_obs, combine_cell_types, bal_combine_bo_covid
from ..utilities import add_obs_cmp_both_label, add_obs_cmp_unique_two
from scipy.stats import pearsonr


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((10, 10), (4, 4))

    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    X = X[~X.obs["patient_category"].isin(["Non-Pneumonia Control", "COVID-19"])]
    combine_cell_types(X)

    cmp1 = 22; cmp2 = 62
    pos1 = True; pos2 = True

    threshold = 0.1
    X = add_obs_cmp_both_label(X, cmp1, cmp2, pos1, pos2, top_perc=threshold)
    X = add_obs_cmp_unique_two(X, cmp1, cmp2)
    
    print(X)
    plot_avegene_scatter_cmps(X, ["MS4A1", "LILRA4"], cmp1, cmp2, ax[0])
    
    return f


def plot_avegene_scatter_cmps(
    X: anndata.AnnData,
    genes: list,  # List of two genes
    cmp1, cmp2, 
    ax,
):
    """Plots average gene expression for two genes across samples, with each sample corresponding to one point.

    Args:
        X (anndata.AnnData): Annotated data matrix.
        genes (list): List of two gene names to compare.
        ax (matplotlib.axes.Axes): Axis to plot on.
    """
    if len(genes) != 2:
        raise ValueError("You must provide exactly two genes.")

    gene1, gene2 = genes

    # Extract gene expression data for both genes
    gene1_data = X[:, gene1].to_df()
    gene2_data = X[:, gene2].to_df()

    # Add metadata columns
    gene1_data["Label"] = X.obs["Label"].values
    gene1_data["sample_id"] = X.obs["sample_id"].values
    gene1_data["Status"] = X.obs["binary_outcome"].values

    gene2_data["Label"] = X.obs["Label"].values
    gene2_data["sample_id"] = X.obs["sample_id"].values
    gene2_data["Status"] = X.obs["binary_outcome"].values

    # Drop rows with missing values
    gene1_data = gene1_data.dropna(subset=["Label", "sample_id", "Status"])
    gene2_data = gene2_data.dropna(subset=["Label", "sample_id", "Status"])

    # Group by Label and sample_id and calculate the average expression
    gene1_avg = gene1_data.groupby(["sample_id", "Label"]).mean().reset_index()
    gene2_avg = gene2_data.groupby(["sample_id", "Label"]).mean().reset_index()

    # Filter for the specific labels for each gene
    gene1_avg = gene1_avg[gene1_avg["Label"] == f"Cmp{cmp1}"]  # Replace "Label1" with the desired label for gene1
    gene2_avg = gene2_avg[gene2_avg["Label"] == f"Cmp{cmp2}"]  # Replace "Label2" with the desired label for gene2

    # Rename columns for clarity
    gene1_avg = gene1_avg.rename(columns={gene1: f"Average {gene1}"})
    gene2_avg = gene2_avg.rename(columns={gene2: f"Average {gene2}"})

    # Merge the two datasets on sample_id
    merged_df = pd.merge(
        gene1_avg[["sample_id", f"Average {gene1}"]],
        gene2_avg[["sample_id", f"Average {gene2}", "Status"]],
        on="sample_id"
    )
    merged_df = merged_df.dropna(subset=[f"Average {gene1}", f"Average {gene2}"])
    
    print(pearsonr(merged_df[f"Average {gene1}"],merged_df[f"Average {gene2}"])[0])

    # Create scatter plot
    sns.scatterplot(
        data=merged_df,
        x=f"Average {gene1}",
        y=f"Average {gene2}",
        hue="Status",  # Color points by Status
        ax=ax,
        # s=100  # Adjust point size
    )
    # print(merged_df)

    # Set axis labels
    ax.set_xlabel(f"Average {gene1} Expression (Label1)")
    ax.set_ylabel(f"Average {gene2} Expression (Label2)")
    ax.set_title("Scatter Plot of Average Gene Expression by Sample")

    return merged_df