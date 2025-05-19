"""Table S1: Summary statistics for metadata correlates."""

import anndata
from .common import getSetup
from ..data_import import meta_raw_df
import pandas as pd
import seaborn as sns
from ..utilities import bal_combine_bo_covid, cell_count_perc_df
from ..data_import import add_obs, combine_cell_types
from .commonFuncs.plotGeneral import rotate_xaxis
from matplotlib.axes import Axes


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((8, 8), (3, 3))

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    combine_cell_types(X)

    all_meta_df = meta_raw_df(X, all=True)
    all_meta_df = bal_combine_bo_covid(all_meta_df)
    categories_to_exclude = ["Non-Pneumonia Control"]
    all_meta_df = all_meta_df[~all_meta_df["patient_category"].isin(categories_to_exclude)]
    meta = "BAL_pct_neutrophils"
    corr_df = all_meta_df[["Status", meta]].dropna()
    
    corr_df = corr_df[corr_df["Status"].isin(["D-C19", "L-C19"])] 
    sns.violinplot(corr_df, x="Status", y=meta, ax=ax[0], hue="Status")
    
    
    X = X[~X.obs["patient_category"].isin(["Non-Pneumonia Control", "Other Viral Pneumonia", "Other Pneumonia"])]
    celltype = ["cell_type", "cell_type"]
    types = ["Cell Count", "Cell Type Percentage"]
    
    axs=1
    for i, celltypes in enumerate(celltype):
        for j, type in enumerate(types):
            celltype_count_perc_df = cell_count_perc_df(X, celltype=celltypes, include_control=False)
            if i == 0:
                t_cells = ["CD4 T cells", "CD8 T cells","CM CD8 T cells", "IFN respons. CD8 T cells", "Tregs"]  
                
            if i == 1:
                t_cells = ["CD4 T cells", "CD8 T cells","CM CD8 T cells", "IFN respons. CD8 T cells", "Tregs"] 
                t_cell_mapping = {
                "CD4 T cells": "CD4 T cells",  # Keep as is
                "CD8 T cells": "CD8 T cells",   # Keep as is
                "CM CD8 T cells": "CD8 T cells",  # Merge into CD8
                "IFN respons. CD8 T cells": "CD8 T cells",  # Merge into CD8
                "Tregs": "CD4 T cells"  # Keep as is
            }
                
            df = celltype_count_perc_df[celltype_count_perc_df["Cell Type"].isin(t_cells)].copy().reset_index(drop=True)
        
            if i == 1:
                print(df)
                df["Cell Type"] = df["Cell Type"].map(t_cell_mapping)
                print(df)
                
                
            df["Cell Type"] = df["Cell Type"].astype(str)
            
            print(df)
            final_df = df.reset_index(drop=True)
            
            sns.boxplot(
                data=final_df,
                x="Cell Type",
                y=type,
                hue="Status",
                # order=celltype,
                showfliers=False,
                ax=ax[axs],
            )
            rotate_xaxis(ax[axs])
            
            axs+=1
            
        
    # plot_cell_count(X, ax[3])


    return f


def plot_cell_count(
    X: anndata.AnnData,
    ax: Axes,
    cond: str = "sample_id",
    status1: str = "binary_outcome",
    status2: str = "patient_category",
):
    """Plots overall cell count."""
    df = X.obs[[cond, status1, status2]].reset_index(drop=True)

    df = bal_combine_bo_covid(df)
    dfCond = (
        df.groupby([cond, "Status"], observed=True)
        .size()
        .reset_index(name="Cell Count")
    )

    sns.barplot(data=dfCond, x="Status", y="Cell Count", hue="Status", ax=ax)
    rotate_xaxis(ax)
