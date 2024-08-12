"""
Figure A4: XXX
"""
import anndata
from pf2.figures.common import getSetup, subplotLabel
from pf2.tensor import correct_conditions

from pf2.data_import import  add_obs
from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr
from .commonFuncs.plotGeneral import rotate_xaxis
from matplotlib.axes import Axes
import anndata
from scipy.cluster.hierarchy import linkage
from matplotlib import colormaps
import networkx as nx

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((12, 12), (1, 1))

    X = read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    X.uns["Pf2_A"] = correct_conditions(X)

    condition_factors_df = pd.DataFrame(
        data=X.uns["Pf2_A"],
        columns=[f"Cmp. {i}" for i in np.arange(1, X.uns["Pf2_A"].shape[1] + 1)],
    )
    
    df = partial_correlation_matrix(condition_factors_df, f)
    df = df.where(np.triu(np.ones(df.shape)).astype(bool))
    df = df.where(np.identity(df.shape[0]) != 1,np.NaN)
    df = df.stack().reset_index()

    df.columns = ["Var1", "Var2", "Weight"]
    df["Weight"] = np.abs(df["Weight"]) 
    df = df.loc[df["Weight"] > 0.5]
    df["Weight"] = (np.max(df["Weight"]) - df["Weight"]) / (np.max(df["Weight"]) - np.min(df["Weight"]))
    
    print(df["Weight"].sum())
    
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    G = nx.from_pandas_edgelist(df=df, source="Var1", target="Var2", edge_attr="Weight", create_using=nx.Graph())
    cmap = plt.cm.plasma
    cmap = plt.cm.Purples
            
    M = G.number_of_edges()
    edge_colors = range(2, M + 2)
    edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
    
    nodes = nx.draw_networkx_nodes(G, nx.circular_layout(G), node_color="lightgrey", label="Legend",
                                   node_size=2500)
    nx.draw_networkx_labels(G, nx.circular_layout(G))
    
    
    edges = nx.draw_networkx_edges(
    G,
    nx.circular_layout(G),
    edge_cmap=cmap,
    width=10,
    # alpha=np.abs(df["Weight"].to_numpy()),
    edge_color=edge_colors,
    # edge_vmin=30
)
    # print(edges)
    
    plt.colorbar(edges)
    
    
    


#     for i in range(M):
#         print(edges)
#         print(edges[1])
#         edges[1].set_alpha(edge_alphas[i])
    
        
    # pc = mpl.collections.PatchCollection(edges, cmap=cmap)
    # pc.set_array(edge_colors)
    
    # plt.colorbar(G, ax=ax[0])
    # plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    # G = nx.from_pandas_edgelist(df=df, source="Var1", target="Var2", edge_attr="Weight", create_using=nx.Graph())
    # node_size = df["Weight"].to_numpy()
    
        
    # # cm = sns.cubehelix_palette(as_cmap=True)
    # # nx.draw_networkx_edges(G, pos=nx.circular_layout(G), node_size=node_size, alpha=np.abs(df["Weight"].to_numpy()), ax=ax[0], label="Legend")
    
    # nx.draw_networkx(G, pos=nx.circular_layout(G), with_labels=True, edge_cmap='Purples',edge_vmin = .5,edge_vmax = 1,  width=np.abs(df["Weight"].to_numpy()), ax=ax[0], label="Legend")
    # columns= [f"Cmp. {i}" for i in np.arange(1, X.uns["Pf2_A"].shape[1] + 1)]
    # ax[0].legend()
    
    
    # nx.draw_networkx(G, pos=nx.spring_layout(G), with_labels=True, edge_cmap='Greys',edge_vmin = .5,edge_vmax = 1, width=np.abs(df["Weight"].to_numpy()), ax=ax[1], label="Legend")
    # columns= [f"Cmp. {i}" for i in np.arange(1, X.uns["Pf2_A"].shape[1] + 1)]
    # ax[1].legend()
    # nx.draw_networkx_labels(G, pos=nx.circular_layout(G), labels=columns)
    # nx.draw_networkx_nodes(G, pos=nx.circular_layout(G), node_size=node_size, ax=ax[0])

    
    


    return f


def partial_correlation_matrix(df: pd.DataFrame, f):
    """Plots partial correlation matrix"""
    cov_DF = df.cov()
    Vi = np.linalg.pinv(cov_DF, hermitian=True) 
    Vi_diag = Vi.diagonal()
    D = np.diag(np.sqrt(1 / Vi_diag))
    pCor = -1 * (D @ Vi @ D) 
    pCor[np.diag_indices_from(pCor)] = 1
    pCorr_DF = pd.DataFrame(pCor, columns=cov_DF.columns, index=cov_DF.columns)
    
    
    # pval = calculate_pvalues(pCorr_DF)
    # print(pval)
    # print(np.shape(pval))
    # cmap = sns.color_palette("vlag", as_cmap=True)
    # f = sns.clustermap(pCorr_DF, robust=True, vmin=-1, vmax=1, 
    #                 #    row_cluster=True, 
    #                 #    col_cluster=True, 
    #                 #    annot=True, 
    #                    cmap=cmap, figsize=(25, 25))
    
    return pCorr_DF
    



def calculate_pvalues(df):
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    
    pvalues = df.copy()
    for r in df.columns:
        for c in df.columns:
            tmp = df[df[r].notnull() & df[c].notnull()]
            pvalues[r][c] = round(pearsonr(tmp[r], tmp[c])[1], 4)
            

    return pvalues