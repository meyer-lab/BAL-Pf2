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
    ax, f = getSetup((15, 9), (3, 6))
    subplotLabel(ax)
    
    
    threshold = 0.56
    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    X.uns["Pf2_A"] = correct_conditions(X)
    
    X.uns["Pf2_A"] -= np.min(X.uns["Pf2_A"], axis=0)
    X.uns["Pf2_A"] += np.median(X.uns["Pf2_A"], axis=0)
    X.uns["Pf2_A"] = np.log(X.uns["Pf2_A"])
    
    condition_factors_df = pd.DataFrame(
        data=X.uns["Pf2_A"],
        columns=[f"Cmp. {i}" for i in np.arange(1, X.uns["Pf2_A"].shape[1] + 1)],
    )
    
    pc_df = partial_correlation_matrix(condition_factors_df)
    
    pc_df = pc_df.sort_values("Weight")
    
    pc_df_top = pc_df.loc[pc_df["Weight"] > threshold]
    pc_df_bot = pc_df.loc[pc_df["Weight"] < -threshold]
    pc_df = pd.concat([pc_df_top, pc_df_bot])
    
    pc_df["Var1"] = pc_df["Var1"].map(lambda x: x.lstrip("Cmp. ")).astype(int)
    pc_df["Var2"] = pc_df["Var2"].map(lambda x: x.lstrip("Cmp. ")).astype(int)
    
    print(pc_df)
    
    for i in range(pc_df.shape[0]):
        cmp1 = pc_df.iloc[i, 0]
        cmp2 = pc_df.iloc[i, 1]
        plot_pair_gene_factors(X, cmp1, cmp2, ax[(3*i)])
        plot_pair_cond_factors(X, cmp1, cmp2, ax[(3*i)+1])
        plot_pair_wp(X, cmp1, cmp2, ax[(3*i)+2], frac=.001)
   
    

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
        
def partial_correlation_matrix(df: pd.DataFrame):
    """Plots partial correlation matrix"""
    cov_DF = df.cov()
    Vi = np.linalg.pinv(cov_DF, hermitian=True) 
    Vi_diag = Vi.diagonal()
    D = np.diag(np.sqrt(1 / Vi_diag))
    pCor = -1 * (D @ Vi @ D) 
    pCor[np.diag_indices_from(pCor)] = 1
    df = pd.DataFrame(pCor, columns=cov_DF.columns, index=cov_DF.columns)
    
    df = df.where(np.triu(np.ones(df.shape)).astype(bool))
    df = df.where(np.identity(df.shape[0]) != 1,np.NaN)
    df = df.stack().reset_index()

    df.columns = ["Var1", "Var2", "Weight"]

    
    return df