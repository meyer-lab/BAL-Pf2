"""
Figure A6b_d
"""

import numpy as np
import anndata
from .common import subplotLabel, getSetup
from .commonFuncs.plotGeneral import rotate_xaxis, plot_avegene_cmps, plot_pair_gene_factors, plot_toppfun
from ..data_import import add_obs, combine_cell_types
from .commonFuncs.plotFactors import bot_top_genes
from .commonFuncs.plotPaCMAP import plot_gene_pacmap, plot_labels_pacmap
from ..utilities import bot_top_genes, add_obs_cmp_both_label, add_obs_cmp_unique_two
import matplotlib.colors as mcolors
import pandas as pd
import seaborn as sns

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((10, 10), (4, 4))

    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    X = X[X.obs["patient_category"] != "Non-Pneumonia Control"] 
    combine_cell_types(X)
    
    cmp1 = 9; cmp2 = 32
    pos1 = True; pos2 = True
    threshold = 0.5
    X = add_obs_cmp_both_label(X, cmp1, cmp2, pos1, pos2, top_perc=threshold)
    X = add_obs_cmp_unique_two(X, cmp1, cmp2)
      
    colors = ["black", "fuchsia", "turquoise", "gainsboro"]
    pal = []
    for i in colors:
        pal.append(mcolors.CSS4_COLORS[i])
        
    plot_labels_pacmap(X, "Label", ax[0], color_key=pal)

    genes1 = bot_top_genes(X, cmp=cmp1, geneAmount=1)
    genes2 = bot_top_genes(X, cmp=cmp2, geneAmount=1)
    genes = np.concatenate([genes1, genes2])

    for i, gene in enumerate(genes):
        plot_gene_pacmap(gene, X, ax[i+1])
        
    plot_pair_gene_factors(X, cmp1, cmp2, ax[7])
        
    X = X[X.obs["Label"] != "Both"] 

    genes1 = bot_top_genes(X, cmp=cmp1, geneAmount=3)
    genes2 = bot_top_genes(X, cmp=cmp2, geneAmount=3)
    genes = np.concatenate([genes1[-3:], genes2[-3:]])
    
    for i, gene in enumerate(genes):
        plot_avegene_cmps(X, gene, ax[i+8], order=["Cmp9", "Cmp32", "NoLabel"])
        rotate_xaxis(ax[i+8])
        
        
    plot_toppfun(cmp1, ax[13])
    plot_pair_cond_factors(X, cmp1, cmp2, ax[14])
 
  

    return f



def plot_pair_cond_factors(
    X, cmp1: int, cmp2: int, ax,
):
    """Plots two condition components weights"""
    factors = np.array(X.uns["Pf2_A"])
    XX = factors
    factors -= np.median(XX, axis=0)
    factors /= np.std(XX, axis=0)
    
    df = pd.DataFrame(factors, columns=[f"Cmp. {i}" for i in range(1, factors.shape[1] + 1)])

    sns.scatterplot(data=df, x=f"Cmp. {cmp1}", y=f"Cmp. {cmp2}", ax=ax)
    ax.set(title="Condition Factors")
    