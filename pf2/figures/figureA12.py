"""
Figure A12:
"""

import seaborn as sns
import pandas as pd
import anndata
import numpy as np
from .common import subplotLabel, getSetup
from ..figures.commonFuncs.plotGeneral import rotate_xaxis, bal_combine_bo_covid, add_obs_label
from .figureA6 import cell_count_perc_df
from .figureA11 import add_obs_cmp_both_label
from ..data_import import add_obs, combine_cell_types
from ..figures.commonFuncs.plotPaCMAP import plot_gene_pacmap
from .commonFuncs.plotFactors import bot_top_genes


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((16, 12), (4, 4))

    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    # print(X)
    # add_obs(X, "binary_outcome")
    # add_obs(X, "patient_category")
    # combine_cell_types(X)

    # cmp1 = 3; cmp2 = 26
    # # pos1 = True; pos2 = True
    # # threshold = 0.5
    # # X = add_obs_cmp_both_label(X, cmp1, cmp2, pos1, pos2, top_perc=threshold)
    # X = add_obs_label(X, cmp1, cmp2)

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
    
    # genes1 = ["CD68", "NAAA", "JAML", "TYROBP"] # Macrophage
    # genes2 = ["APOBEC3A", "LYZ", "CD14", "CFP", "HLA-DRA", "S100A9",
    #          "S100A8", "CSF3R", "FCGR3A"] # Monocyte
    # genes1 = ["FCGR3A"]  # nCM
    # genes2 = ["CSF3R", "S100A8", "TREM1", "IL1R2", "CFP", "ADAM8"] # Neutrophil
    # genes3 = ["S100A4", "S100A9", "ICAM1", "S100A8", "ITGAM"] # Myeloid supressor 
    # genes1 = ["STMN1"] # Prolif
    # genes2 = ["MZB1", "SPAG4"] # Plasma
    # genes3 = ["TRBC2", "CD3D", "CD3G", "CD3E", "LTB", "IL7R", "LEF1"] # T cell
    # genes1 = ["PXK", "MS4A1", "CD19", "CD74", "CD79A", "BANK1", "PTPRC"] # B cell
    # genes2 = ["TRAC", "CD8A", "GZMB", "CD2", "CD27", "CD5", "CD27"] # Cytotoxic T cell
    # genes1 = ["IKZF2", "FOXP3", "CCR4", "ENTPD1", "IL2RA", "ITGAE", "TNFRSF4", "CTLA4"] # Follicular helper T cell
    # genes2 = ["CCR4", "CD4", "CD28", "CD3G", "CCR6"] # Helper T cel
    # genes1 = ["CCR7", "CD2", "PTPRC", "CD28", "LEF1", "S100A8", "GIMAP4"] # Memory T cell
    
    # genes1 = ["NKG7", "GNLY", "KLRD1", "KLRF1",  "DOCK2", "GZMA"] # NK
    
    genes1 = ["FOXJ1", "CCDC78", "MUC5AC", "MUC5B"] # Ciliated
    genes2 = ["TPM1"] # ionocytes
    genes3 = ["SCGB1A1", "BPIFA1", "SCGB3A2"] # Secretory cells (also have MUC5AC and MUC5B)
    genes = np.concatenate([genes1, genes2, genes3])

    # genes = ["NEK2", "KIF20A", "RAD54L",  "FAM111B"]

    for i, gene in enumerate(genes):
        plot_gene_pacmap(gene, X, ax[i])
    
    # for i, gene in enumerate(np.ravel(genes)):
    #     plot_avegene_per_status(X, gene, ax[i+4], cellType="leiden", othercelltype="Label")
    #     rotate_xaxis(ax[i])
    
    # for i, gene in enumerate(np.ravel(genes)):
    #     plot_avegene_per_status(X, gene, ax[i], cellType="leiden")
    #     rotate_xaxis(ax[i])

    return f

def plot_avegene_per_status(
    X: anndata.AnnData,
    gene: str,
    ax,
    condition="sample_id",
    cellType="cell_type",
    status1="binary_outcome",
    status2="patient_category",
):
    """Plots average gene expression across cell types for a category of drugs"""
    genesV = X[:, gene]
    dataDF = genesV.to_df()
    dataDF = dataDF.subtract(genesV.var["means"].values)
    dataDF[status1] = genesV.obs[status1].values
    dataDF[status2] = genesV.obs[status2].values
    dataDF["Condition"] = genesV.obs[condition].values
    dataDF["Cell Type"] = genesV.obs[cellType].values


    df = bal_combine_bo_covid(dataDF, status1, status2)

    df = pd.melt(
        df, id_vars=["Status", "Cell Type", "Condition"], value_vars=gene
    ).rename(columns={"variable": "Gene", "value": "Value"})

    df = df.groupby(["Status", "Cell Type", "Gene", "Condition"], observed=False).mean()
    df = df.rename(columns={"Value": "Average Gene Expression"}).reset_index()

    sns.boxplot(
        data=df.loc[df["Gene"] == gene],
        x="Cell Type",
        y="Average Gene Expression",
        hue="Status",
        ax=ax,
        showfliers=False,
    )
    ax.set(ylabel=f"Average {gene}")

    return df

# def plot_avegene_per_status(
#     X: anndata.AnnData,
#     gene: str,
#     ax,
#     condition="sample_id",
#     cellType="cell_type",
#     status1="binary_outcome",
#     status2="patient_category",
#     othercelltype="leiden",
# ):
#     """Plots average gene expression across cell types for a category of drugs"""
#     genesV = X[:, gene]
#     dataDF = genesV.to_df()
#     dataDF = dataDF.subtract(genesV.var["means"].values)
#     dataDF[status1] = genesV.obs[status1].values
#     dataDF[status2] = genesV.obs[status2].values
#     dataDF["Condition"] = genesV.obs[condition].values
#     dataDF["Cell Type"] = genesV.obs[cellType].values
#     dataDF["Other"] = genesV.obs[othercelltype].values

#     df = bal_combine_bo_covid(dataDF, status1, status2)

#     df = pd.melt(
#         df, id_vars=["Other", "Cell Type", "Condition"], value_vars=gene
#     ).rename(columns={"variable": "Gene", "value": "Value"})

#     df = df.groupby(["Other", "Cell Type", "Gene", "Condition"], observed=False).mean()
#     df = df.rename(columns={"Value": "Average Gene Expression"}).reset_index()

#     sns.boxplot(
#         data=df.loc[df["Gene"] == gene],
#         x="Cell Type",
#         y="Average Gene Expression",
#         hue="Other",
#         ax=ax,
#         showfliers=False,
#     )
#     ax.set(ylabel=f"Average {gene}")

#     return df