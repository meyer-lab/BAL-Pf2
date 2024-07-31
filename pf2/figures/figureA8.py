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


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((8, 8), (4, 4))
    subplotLabel(ax)
    
    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    X.uns["Pf2_A"] = correct_conditions(X)
    
    plot_pair_gene_factors(X, 25, 49, ax[0])
    plot_pair_cond_factors(X, 25, 29, ax[1])
    plot_pair_wp(X, 25, 49, ax[2], frac=.001)
    plot_pair_wp_pacmap(X, 24, 49, ax[3])
    
    
    plot_pair_gene_factors(X, 28, 45, ax[4])
    plot_pair_cond_factors(X, 28, 45, ax[5])
    plot_pair_wp(X, 28, 45, ax[6], frac=.001)
    plot_pair_wp_pacmap(X, 38, 45, ax[7])
    
    ax[0].set(xlim=(-.05, .15), ylim=(-.05, .15))
    ax[1].set(xlim=(-1, 10), ylim=(-1, 20))
    ax[2].set(xlim=(-.15, .15), ylim=(-.1, .15))
    
   
    ax[4].set(xlim=(-.3, .4), ylim=(-.4, .4))
    ax[5].set(xlim=(-1, 20), ylim=(-1, 20))
    ax[6].set(xlim=(-.4, .4), ylim=(-.4, .2))
    
    plot_y_x_line(ax[0], pos=True)
    plot_y_x_line(ax[1], pos=True)
    plot_y_x_line(ax[2], pos=False)
    
    plot_y_x_line(ax[4], pos=True)
    plot_y_x_line(ax[5], pos=True)
    plot_y_x_line(ax[6], pos=False)
    
    
    # ax[0].set(xlim=(-.5, .15), ylim=(-.5, .15))
    # ax[1].set(xlim=(-1, 20), ylim=(-1, 20))
    
    
    # ax[0].set(xticks=[-.4, 0, 4], yticks=[-.4, 0, .4])
    # # ax[1].set(xticks=[-1, 5, 10, 15, 20, 25], yticks=[-1, 5, 10, 15, 20, 25])
    # ax[1].set(xlim=(-2, 25), ylim=(-2, 25))

    return f




def plot_pair_gene_factors(X: anndata.AnnData, cmp1: int, cmp2: int, ax: Axes):
    """Plots two gene components weights"""
    cmpWeights = np.concatenate(
        ([X.varm["Pf2_C"][:, cmp1 - 1]], [X.varm["Pf2_C"][:, cmp2 - 1]])
    )
    df = pd.DataFrame(
        data=cmpWeights.transpose(), columns=[f"Cmp. {cmp1}", f"Cmp. {cmp2}"]
    )
    sns.scatterplot(data=df, x=f"Cmp. {cmp1}", y=f"Cmp. {cmp2}", ax=ax)
    ax.set(title="Gene Factors")
    
    
    

def plot_pair_cond_factors(X: anndata.AnnData, cmp1: int, cmp2: int, ax: Axes):
    """Plots two gene components weights"""
    cmpWeights = np.concatenate(
        ([X.uns["Pf2_A"][:, cmp1 - 1]], [X.uns["Pf2_A"][:, cmp2 - 1]])
    )
    df = pd.DataFrame(
        data=cmpWeights.transpose(), columns=[f"Cmp. {cmp1}", f"Cmp. {cmp2}"]
    )
    sns.scatterplot(data=df, x=f"Cmp. {cmp1}", y=f"Cmp. {cmp2}", ax=ax)
    ax.set(title="Condition Factors")
    
    
def plot_pair_wp(X: anndata.AnnData, cmp1: int, cmp2: int, ax: Axes, frac: float = .001):
    """Plots two gene components weights"""
    cmpWeights = np.concatenate(
        ([X.obsm["weighted_projections"][:, cmp1 - 1]], [X.obsm["weighted_projections"][:, cmp2 - 1]])
    )
    df = pd.DataFrame(
        data=cmpWeights.transpose(), columns=[f"Cmp. {cmp1}", f"Cmp. {cmp2}"]
    )
    df = df.sample(frac=frac)
    
    sns.scatterplot(data=df, x=f"Cmp. {cmp1}", y=f"Cmp. {cmp2}", ax=ax)
    ax.set(title=f"WP {frac*100}% of Cells")


def plot_y_x_line(ax: Axes, pos=True): 
    lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]), 
    np.max([ax.get_xlim(), ax.get_ylim()])]
    if pos is True:  
        ax.plot(lims, lims, 'k-')
    else:
        ax.plot(lims, np.flip(lims), 'k-')
        
        print(lims)