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
    
    # genes = ["CD68", "NAAA", "JAML", "TYROBP"] # Macrophage
    # genes = ["APOBEC3A", "LYZ", "CD14", "CFP", "HLA-DRA", "S100A9",
    #          "S100A8", "CSF3R", "FCGR3A"] # Monocyte
    # genes = ["FCGR3A"]  # nCM
    
    # genes = ["CSF3R", "S100A8", "TREM1", "IL1R2", "CFP", "ADAM8"] # Neutrophil
    # genes = ["S100A4", "S100A9", "ICAM1", "S100A8", "ITGAM"] # Myeloid supressor 
    genes = ["STMN1"] # Prolif
    genes = ["MZB1", "SPAG4"] # Plasma
    # genes = ["TRBC2", "CD3D", "CD3G", "CD3E", "LTB", "IL7R", "LEF1"] # T cell
    genes = ["PXK", "MS4A1", "CD19", "CD74", "CD79A", "BANK1", "PTPRC"] # B cell
    # genes = ["TRAC", "CD8A", "GZMB", "CD2", "CD27", "CD5", "CD27"] # Cytotoxic T cell
    # genes = ["IKZF2", "FOXP3", "CCR4", "ENTPD1", "IL2RA", "ITGAE", "TNFRSF4", "CTLA4"] # Follicular helper T cell
    # genes = ["CCR4", "CD4", "CD28", "CD3G", "CCR6"] # Helper T cel
    # genes = ["CCR7", "CD2", "PTPRC", "CD28", "LEF1", "S100A8", "GIMAP4"] # Memory T cell
    
    # genes = ["NKG7", "GNLY", "KLRD1", "KLRF1",  "DOCK2", "GZMA"] # NK
    
    genes = ["FOXJ1", "CCDC78", "MUC5AC", "MUC5B"] # Ciliated
    genes = ["GPRC5B", "SLC4A9"] # ionocytes

    for i, gene in enumerate(genes):
        plot_gene_pacmap(gene, X, ax[i + 2])

    return f
