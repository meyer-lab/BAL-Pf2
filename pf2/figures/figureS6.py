"""
Figure S6:
"""

import seaborn as sns
import pandas as pd
import anndata
import numpy as np
from .common import subplotLabel, getSetup
from .commonFuncs.plotGeneral import rotate_xaxis
from .figureA1d_g import cell_count_perc_df
from .figureA5b_i import add_obs_cmp_both_label
from ..data_import import add_obs, combine_cell_types
from .commonFuncs.plotPaCMAP import plot_gene_pacmap
from .commonFuncs.plotFactors import bot_top_genes


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((12, 12), (4, 4))

    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    combine_cell_types(X)
    X = X[X.obs["patient_category"] != "Non-Pneumonia Control"]

    cmp1 = 27; cmp2 = 46
    pos1 = True; pos2 = True
    threshold = 0.5
    X = add_obs_cmp_both_label(X, cmp1, cmp2, pos1, pos2, top_perc=threshold)

    celltype_count_perc_df_1 = cell_count_perc_df(
        X[(X.obs[f"Cmp{cmp1}"] == True) & (X.obs["Both"] == False)],
        celltype="combined_cell_type",
    )
    celltype_count_perc_df_1["Label"] = f"Cmp{cmp1}"
    celltype_count_perc_df_2 = cell_count_perc_df(
        X[(X.obs[f"Cmp{cmp2}"] == True) & (X.obs["Both"] == False)],
        celltype="combined_cell_type",
    )
    celltype_count_perc_df_2["Label"] = f"Cmp{cmp2}"
    celltype_count_perc_df_3 = cell_count_perc_df(
        X[X.obs["Both"] == True], celltype="combined_cell_type"
    )
    celltype_count_perc_df_3["Label"] = "Both"
    celltype_count_perc_df = pd.concat(
        [
            celltype_count_perc_df_1,
            celltype_count_perc_df_2,
            celltype_count_perc_df_3,
        ],
        axis=0,
    )

    hue = ["Cell Type", "Status"]
    for i in range(2):
        sns.boxplot(
            data=celltype_count_perc_df,
            x="Label",
            y="Cell Count",
            hue=hue[i],
            showfliers=False,
            ax=ax[i],
        )
        rotate_xaxis(ax[i])

    return f
