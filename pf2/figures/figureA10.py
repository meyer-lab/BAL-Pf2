"""
XXXX
"""

import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.axes import Axes
import anndata
from .common import subplotLabel, getSetup
from .commonFuncs.plotPaCMAP import plot_wp_pacmap, plot_pair_wp_pacmap
from ..tensor import correct_conditions
from ..data_import import condition_factors_meta, combine_cell_types, add_obs
from pf2.figures.commonFuncs.plotGeneral import rotate_xaxis, bal_combine_bo_covid, avegene_per_status

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((18, 16), (5, 4))
    subplotLabel(ax)
    
    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    
    combine_cell_types(X)
    add_obs(X, "patient_category")
    add_obs(X, "binary_outcome")
  


    genes = ["EDN1", "PADI4"]
    df_total = pd.DataFrame([])
    for i, gene in enumerate(np.ravel(genes)):
        df = avegene_per_status(X, gene, cellType="combined_cell_type")
        df_total = pd.concat([df, df_total])

    plot_ave2genes_per_status(df_total, genes[0], genes[1], ax[0])

    return f

    

def plot_ave2genes_per_status(df_total, gene1, gene2, ax):
    """Plots average of 2 genes per celltype per status"""
    df_total = df_total.pivot(
        index=["Status", "Cell Type", "Condition"],
        columns="Gene",
        values="Average Gene Expression",
    )
    df_mean = (
        df_total.groupby(["Status", "Cell Type"], observed=False)
        .mean()
        .dropna()
        .reset_index()
    )
    df_std = (
        df_total.groupby(["Status", "Cell Type"], observed=False)
        .std()
        .dropna()
        .reset_index()
    )

    colors = sns.color_palette("hls", len(np.unique(df_mean["Cell Type"])))
    fmt = ["o", "*"]

    for i, status in enumerate(np.unique(df_mean["Status"])):
        for j, celltype in enumerate(np.unique(df_mean["Cell Type"])):
            df_mini_mean = df_mean.loc[
                (df_mean["Status"] == status) & (df_mean["Cell Type"] == celltype)
            ]
            df_mini_std = df_std.loc[
                (df_std["Status"] == status) & (df_std["Cell Type"] == celltype)
            ]
            ax.errorbar(
                df_mini_mean[gene1],
                df_mini_mean[gene2],
                xerr=df_mini_std[gene1],
                yerr=df_mini_std[gene2],
                ls="none",
                fmt=fmt[i],
                label=celltype + status,
                color=colors[j],
            )

    ax.set(xlabel=f"Average {gene1}", ylabel=f"Average {gene2}")
    ax.legend()
