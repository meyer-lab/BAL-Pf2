"""
Figure A6h_m: Analysis of component correlations, gene expression (SFN, AGER, SCGB3A2), and cell type percentages
"""

import anndata
from .common import subplotLabel, getSetup
from .commonFuncs.plotGeneral import rotate_xaxis,  plot_avegene_cmps, plot_pair_gene_factors
from ..data_import import add_obs, combine_cell_types
from ..utilities import add_obs_cmp_both_label, add_obs_cmp_unique_two
from RISE.figures.commonFuncs.plotPaCMAP import plot_labels_pacmap
import matplotlib.colors as mcolors
from ..utilities import cell_count_perc_df
import seaborn as sns
import pandas as pd

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((10, 10), (3, 3))

    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    X = X[~X.obs["patient_category"].isin(["Non-Pneumonia Control", "COVID-19"])]
    combine_cell_types(X)

    cmp1 = 55; cmp2 = 67
    pos1 = True; pos2 = True
    threshold = 0.1
    X = add_obs_cmp_both_label(X, cmp1, cmp2, pos1, pos2, top_perc=threshold)
    X = add_obs_cmp_unique_two(X, cmp1, cmp2)
    
    plot_pair_gene_factors(X, cmp1, cmp2, ax[0])
      
    colors = ["black",  "turquoise", "fuchsia", "gainsboro"]
    pal = []
    for i in colors:
        pal.append(mcolors.CSS4_COLORS[i])
        
    plot_labels_pacmap(X, "Label", ax[1], color_key=pal)

    X = X[X.obs["Label"] != "Both"] 
    
    add_obs(X, "immunocompromised_flag")
    plot_avegene_scatter_cmps(X, ["SFN", "AGER"], cmp1, cmp1, ax[2], otherlabel="immunocompromised_flag")

    genes = ["SFN", "SCGB3A2"]
    for i, gene in enumerate(genes):
        plot_avegene_cmps(X, gene, ax[i+3])
        rotate_xaxis(ax[i+3])
        
    celltype = "combined_cell_type"
    type = "Cell Type Percentage"
    

    celltype_count_perc_df = cell_count_perc_df(X, celltype=celltype, include_control=False)
    new_df = celltype_count_perc_df.loc[celltype_count_perc_df["Cell Type"] == "Other"].copy().reset_index(drop=True)
    new_df["Cell Type"] = new_df["Cell Type"].astype(str)
    final_df = new_df.reset_index(drop=True)

    sns.boxplot(
        data=final_df,
        x="Cell Type",
        y=type,
        hue="Status",
        showfliers=False,
        dodge=True,
        gap=.1,
        ax=ax[5],
    )
    rotate_xaxis(ax[5])

    return f



def plot_avegene_scatter_cmps(
    X: anndata.AnnData,
    genes: list,  # List of two genes
    cmp1, cmp2, 
    ax,
    otherlabel="Label",
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
    gene1_data[otherlabel] = X.obs[otherlabel].values

    gene2_data["Label"] = X.obs["Label"].values
    gene2_data["sample_id"] = X.obs["sample_id"].values
    gene2_data["Status"] = X.obs["binary_outcome"].values
    gene2_data[otherlabel] = X.obs[otherlabel].values

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
        gene2_avg[["sample_id", f"Average {gene2}", "Status", otherlabel]],
        on="sample_id"
    )
    merged_df = merged_df.dropna(subset=[f"Average {gene1}", f"Average {gene2}"])
    merged_df["Status"] = merged_df["Status"].replace({1: "D-nC19", 0: "L-nC19"})

    pal = sns.color_palette()
    pal = [pal[1], pal[3]]
    pal = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' for r, g, b in pal]
    sns.scatterplot(
        data=merged_df,
        x=f"Average {gene1}",
        y=f"Average {gene2}",
        hue="Status",  
        palette=pal,
        style=otherlabel,
        ax=ax,
    )

    ax.set_xlabel(f"Average {gene1} Expression")
    ax.set_ylabel(f"Average {gene2} Expression")
    ax.set(xlim=(-.05, .32), ylim=(-.05, .32))

    return merged_df