"""
Figure S7:
"""

import seaborn as sns
import pandas as pd
import anndata
import numpy as np
from .common import subplotLabel, getSetup
from .commonFuncs.plotGeneral import rotate_xaxis
from .figureA1d_g import cell_count_perc_df
from .commonFuncs.plotGeneral import rotate_xaxis, plot_avegene_cmps, add_obs_cmp_both_label_three, add_obs_label_three  
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
    X = X[X.obs["patient_category"] != "Non-Pneumonia Control"]
    combine_cell_types(X)

   
    cmp1 = 20; cmp2 = 27; cmp3 = 35
    pos1 = True; pos2 = True; pos3 = False
    threshold = 0.5
    X = add_obs_cmp_both_label_three(X, cmp1, cmp2, cmp3, pos1, pos2, pos3, top_perc=threshold)
    X = add_obs_label_three(X, cmp1, cmp2, cmp3)

    celltype_count_perc_df_1 = cell_count_perc_df(
        X[(X.obs[f"Cmp{cmp1}"] == True) & (X.obs[f"Cmp{cmp2}"] == False) &  (X.obs[f"Cmp{cmp3}"] == False) & (X.obs["Both"] == False)],
        celltype="combined_cell_type",
    )
    celltype_count_perc_df_1["Label"] = f"Cmp{cmp1}"
    celltype_count_perc_df_2 = cell_count_perc_df(
        X[(X.obs[f"Cmp{cmp1}"] == False) & (X.obs[f"Cmp{cmp2}"] == True) &  (X.obs[f"Cmp{cmp3}"] == False) & (X.obs["Both"] == False)],
        celltype="combined_cell_type",
    )
    celltype_count_perc_df_2["Label"] = f"Cmp{cmp2}"
    celltype_count_perc_df_3 = cell_count_perc_df(
            X[(X.obs[f"Cmp{cmp1}"] == False) & (X.obs[f"Cmp{cmp2}"] == False) &  (X.obs[f"Cmp{cmp3}"] == True) & (X.obs["Both"] == False)], 
            celltype="combined_cell_type"
    )
    celltype_count_perc_df_3["Label"] = f"Cmp{cmp3}"
    
    celltype_count_perc_df_4 = cell_count_perc_df(
            X[(X.obs[f"Cmp{cmp1}"] == True) & (X.obs[f"Cmp{cmp2}"] == True) &  (X.obs[f"Cmp{cmp3}"] == True) & (X.obs["Both"] == True)], 
            celltype="combined_cell_type"
    )
    celltype_count_perc_df_4["Label"] = f"Both"



    celltype_count_perc_df = pd.concat(
        [
            celltype_count_perc_df_1,
            celltype_count_perc_df_2,
            celltype_count_perc_df_3,
            celltype_count_perc_df_4
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
