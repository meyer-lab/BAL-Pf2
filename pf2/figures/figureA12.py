"""
Figure A12:
"""

import seaborn as sns
import pandas as pd
import anndata
import numpy as np
from .common import subplotLabel, getSetup
from ..figures.commonFuncs.plotGeneral import rotate_xaxis
from .figureA6 import cell_count_perc_df
from .figureA11 import add_obs_cmp_both_label
from ..data_import import add_obs, combine_cell_types
from ..figures.commonFuncs.plotPaCMAP import plot_gene_pacmap
from .commonFuncs.plotFactors import bot_top_genes


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((12, 12), (4, 4))

    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    # add_obs(X, "binary_outcome")
    # add_obs(X, "patient_category")
    # combine_cell_types(X)

    # cmp1 = 27; cmp2 = 46
    # pos1 = True; pos2 = True
    # threshold = 0.5
    # X = add_obs_cmp_both_label(X, cmp1, cmp2, pos1, pos2, top_perc=threshold)

    # celltype_count_perc_df_1 = cell_count_perc_df(
    #     X[(X.obs[f"Cmp{cmp1}"] == True) & (X.obs["Both"] == False)],
    #     celltype="combined_cell_type",
    # )
    # celltype_count_perc_df_1["Label"] = f"Cmp{cmp1}"
    # celltype_count_perc_df_2 = cell_count_perc_df(
    #     X[(X.obs[f"Cmp{cmp2}"] == True) & (X.obs["Both"] == False)],
    #     celltype="combined_cell_type",
    # )
    # celltype_count_perc_df_2["Label"] = f"Cmp{cmp2}"
    # celltype_count_perc_df_3 = cell_count_perc_df(
    #     X[X.obs["Both"] == True], celltype="combined_cell_type"
    # )
    # celltype_count_perc_df_3["Label"] = "Both"
    # celltype_count_perc_df = pd.concat(
    #     [
    #         celltype_count_perc_df_1,
    #         celltype_count_perc_df_2,
    #         celltype_count_perc_df_3,
    #     ],
    #     axis=0,
    # )

    # hue = ["Cell Type", "Status"]
    # for i in range(2):
    #     sns.boxplot(
    #         data=celltype_count_perc_df,
    #         x="Label",
    #         y="Cell Count",
    #         hue=hue[i],
    #         showfliers=False,
    #         ax=ax[i],
    #     )
    #     rotate_xaxis(ax[i])

    # genes1 = bot_top_genes(X, cmp=cmp1, geneAmount=1)
    # genes2 = bot_top_genes(X, cmp=cmp2, geneAmount=1)
    # genes = np.concatenate([genes1, genes2])
    
    genes = ["CD68", "NAAA", "JAML", "TYROBP"]
    # genes = ["APOBEC3A", "LYZ", "CD14", "CFP", "HLA-DRA", "S100A9",
    #          "S100A8", "CSF3R"]

    for i, gene in enumerate(genes):
        plot_gene_pacmap(gene, X, ax[i + 2])

    return f
