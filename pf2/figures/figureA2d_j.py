"""Figure A2d_j"""

import anndata
import pandas as pd
from ..data_import import combine_cell_types, add_obs
from .common import getSetup, subplotLabel
from .commonFuncs.plotFactors import (
    plot_condition_factors,
    plot_gene_factors,
    plot_eigenstate_factors,
)
from .commonFuncs.plotPaCMAP import plot_labels_pacmap
from ..data_import import condition_factors_meta, bal_combine_bo_covid
import seaborn as sns


def makeFigure():
    ax, f = getSetup((20, 20), (3, 3))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    meta_info = ["patient_category", "binary_outcome", "episode_etiology", "episode_category"]
    for i in meta_info:
        add_obs(X, i)   

    cond_fact_meta_df = condition_factors_meta(X)
    
    pal = sns.color_palette()
    pal = [pal[0], pal[5], pal[1], pal[2], pal[4], pal[3]]
    pal = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' for r, g, b in pal]
    
    plot_condition_factors(
        X, ax[0], cond="sample_id", cond_group_labels=pd.Series(cond_fact_meta_df["Status"]), color_key=pal, group_cond=True)
    ax[0].yaxis.set_ticklabels([])
    
    # plot_eigenstate_factors(X, ax[1])
    # plot_gene_factors(X, ax[2])
    # ax[2].yaxis.set_ticklabels([])
    
    # X = X[X.obs["patient_category"] != "Non-Pneumonia Control"] 
    # df = X.obs[["patient_category", "binary_outcome"]].reset_index(drop=True)
    # df = bal_combine_bo_covid(df)
    # X.obs["Status"] = df["Status"].to_numpy()
    # plot_labels_pacmap(X, "Status", ax[3], color_key=pal)
    
    # combine_cell_types(X)
    # plot_labels_pacmap(X, "cell_type", ax[4]) 
    
    # pal = sns.color_palette(palette='Set3')
    # pal = pal.as_hex() 
    # plot_labels_pacmap(X, "combined_cell_type", ax[5], color_key=pal)
    
    # pal = sns.color_palette(palette='cubehelix')
    # pal = pal.as_hex() 
    # XX = X[~X.obs["episode_etiology"].isna()]
    # plot_labels_pacmap(XX, "episode_etiology", ax[6], color_key=pal)
    
    # pal = sns.color_palette(palette='CMRmap')
    # pal = pal.as_hex() 
    # XX = X[~X.obs["episode_category"].isna()]
    # plot_labels_pacmap(XX, "episode_category", ax[7], color_key=pal)

    return f

