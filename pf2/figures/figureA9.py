"""
Figure A4: XXX
"""
import anndata
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from ..tensor import correct_conditions
from .common import getSetup

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((12, 12), (1, 1))

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    X.uns["Pf2_A"] = correct_conditions(X)

    condition_factors_df = pd.DataFrame(
        data=X.uns["Pf2_A"],
        columns=[f"Cmp. {i}" for i in np.arange(1, X.uns["Pf2_A"].shape[1] + 1)],
    )
    
    pc_df = partial_correlation_matrix(condition_factors_df)
    
    plot_network_graph(pc_df)


    return f


def partial_correlation_matrix(df: pd.DataFrame):
    """Plots partial correlation matrix"""
    cov_DF = df.cov()
    Vi = np.linalg.pinv(cov_DF, hermitian=True) 
    Vi_diag = Vi.diagonal()
    D = np.diag(np.sqrt(1 / Vi_diag))
    pCor = -1 * (D @ Vi @ D) 
    pCor[np.diag_indices_from(pCor)] = 1
    pCorr_DF = pd.DataFrame(pCor, columns=cov_DF.columns, index=cov_DF.columns)

    
    return pCorr_DF
    
    
def plot_network_graph(df: pd.DataFrame, threshold: float = 0.5, cmap = plt.cm.plasma):
    """Plots network for correlation matrix """
    df = df.where(np.triu(np.ones(df.shape)).astype(bool))
    df = df.where(np.identity(df.shape[0]) != 1,np.NaN)
    df = df.stack().reset_index()

    df.columns = ["Var1", "Var2", "Weight"]
    df["Weight"] = np.abs(df["Weight"]) 
    df = df.loc[df["Weight"] > threshold]
  
    G = nx.from_pandas_edgelist(df=df, source="Var1", target="Var2", edge_attr="Weight", create_using=nx.Graph()) 
    M = G.number_of_edges()
    edge_colors = range(2, M + 2)
    
    nx.draw_networkx_nodes(G, nx.circular_layout(G), node_color="lightgrey", node_size=2500)
    nx.draw_networkx_labels(G, nx.circular_layout(G))
    
    edges = nx.draw_networkx_edges(
    G,
    nx.circular_layout(G),
    edge_cmap=cmap,
    width=10,
    edge_color=edge_colors)
    
    plt.colorbar(edges)
