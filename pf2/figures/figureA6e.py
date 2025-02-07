"""
Figure 6e
"""

import numpy as np
import anndata
from .common import subplotLabel, getSetup
from .commonFuncs.plotGeneral import rotate_xaxis, add_obs_cmp_both_label, add_obs_label, plot_avegene_cmps, plot_pair_gene_factors, bal_combine_bo_covid
from ..data_import import add_obs, combine_cell_types, condition_factors_meta
from .commonFuncs.plotFactors import bot_top_genes
from .commonFuncs.plotPaCMAP import plot_gene_pacmap, plot_labels_pacmap, plot_wp_pacmap
import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sns


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # ax, f = getSetup((10, 10), (4, 4))
    ax, f = getSetup((6, 12), (6, 3))

    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    # X = X[X.obs["patient_category"] != "Non-Pneumonia Control"] 
    combine_cell_types(X)

    cmp1 = 28; cmp2 = 38
    pos1 = True; pos2 = True
    threshold = 0.1
    X = add_obs_cmp_both_label(X, cmp1, cmp2, pos1, pos2, top_perc=threshold)
    X = add_obs_label(X, cmp1, cmp2)
      
    colors = ["black",  "turquoise", "fuchsia", "gainsboro"]
    # pal = []
    # for i in colors:
    #     pal.append(mcolors.CSS4_COLORS[i])
        
    # plot_labels_pacmap(X, "Label", ax[0], color_key=pal)

    genes1 = bot_top_genes(X, cmp=cmp1, geneAmount=2)
    genes2 = bot_top_genes(X, cmp=cmp2, geneAmount=2)
    genes = np.concatenate([genes1, genes2])

    # for i, gene in enumerate(genes):
    #     plot_gene_pacmap(gene, X, ax[i+1])
        
    # for i, cmp in enumerate([cmp1, cmp2]):
    #     plot_wp_pacmap(X, cmp, ax[i+5], cbarMax=0.4)
        
    # plot_pair_gene_factors(X, cmp1, cmp2, ax[7])
    # plot_pair_gene_factors(X, cmp1, 45, ax[0])
    # plot_pair_cond_factors(X,  cmp1, 45, ax[1])
        
    X = X[X.obs["Label"] != "Both"] 

    for i, gene in enumerate(genes):
        plot_avegene_cmps(X, gene, ax[i])
        rotate_xaxis(ax[i])
    
    # plot_pair_cond_factors(X, cmp1, cmp2, ax[0])
 
  

    return f



def plot_pair_cond_factors(
    X, cmp1: int, cmp2: int, ax,
):
    """Plots two condition components weights"""
    factors = np.array(X.uns["Pf2_A"])
    XX = factors
    
    cond_fact_meta_df = condition_factors_meta(X)
    cond_fact_meta_df = cond_fact_meta_df[cond_fact_meta_df["patient_category"] != "Non-Pneumonia Control"]
    cond_fact_meta_df = bal_combine_bo_covid(cond_fact_meta_df)
    cond_fact_meta_df = cond_fact_meta_df[cond_fact_meta_df["Status"] != "D-C19"]
    cond_fact_meta_df = cond_fact_meta_df[cond_fact_meta_df["Status"] != "L-C19"]
    cond_fact_meta_df = cond_fact_meta_df[[f"Cmp. {cmp1}", f"Cmp. {cmp2}"]]
    # factors -= np.median(XX, axis=0)
    # factors /= np.std(XX, axis=0)
    
    # df = pd.DataFrame(factors, columns=[f"Cmp. {i}" for i in range(1, factors.shape[1] + 1)])
    
    print(cond_fact_meta_df[f"Cmp. {cmp1}"].corr(cond_fact_meta_df[f"Cmp. {cmp2}"]))

    sns.scatterplot(data=cond_fact_meta_df, x=f"Cmp. {cmp1}", y=f"Cmp. {cmp2}", ax=ax, color="k")
    ax.set(title="Condition Factors")
    