from anndata import AnnData
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    
def plot_gene_factors_defined(
    cmps: list, dataIn: AnnData, ax: Axes, geneAmount: int = 5
):
    """Plotting weights for gene factors for both most negatively/positively weighted terms"""
    cmpName = [f"Cmp. {cmp}" for cmp in cmps]
    cmp_idx = [cmp - 1 for cmp in cmps]
    df = pd.DataFrame(
        data=dataIn.varm["Pf2_C"][:, cmp_idx], index=dataIn.var_names, columns=[f"Cmp. {cmp}" for cmp in cmps]
    )
    genes = []
    for i, cmp in enumerate(cmpName):
        df_sorted = df.iloc[:, i].reset_index(drop=False)
        df_sorted = df_sorted.sort_values(by=cmp).set_index("index")  
        weighted_genes = np.concatenate((df_sorted.index[:geneAmount].values, df_sorted.index[-geneAmount:].values))
        genes = np.concatenate((genes, weighted_genes))

    df = df.loc[genes]
    df.drop_duplicates(inplace=True)
    df = df.div(np.max(np.abs(df), axis=0).values)
    
    sns.heatmap(
        data=df,
        ax=ax,
        center=0,
        cmap=cmap,
        vmin=-1,
        vmax=1,
    )
    ax.tick_params(axis="x", rotation=90)
    

def plot_gene_factors_partial(
    cmp: int, dataIn: AnnData, ax: Axes, geneAmount: int = 5, top=True
):
    """Plotting weights for gene factors for both most negatively/positively weighted terms"""
    cmpName = f"Cmp. {cmp}"

    df = pd.DataFrame(
        data=dataIn.varm["Pf2_C"][:, cmp - 1], index=dataIn.var_names, columns=[cmpName]
    )
    df = df.reset_index(names="Gene")
    df = df.sort_values(by=cmpName)

    if top:
        sns.barplot(
            data=df.iloc[-geneAmount:, :], x="Gene", y=cmpName, color="k", ax=ax
        )
    else:
        sns.barplot(data=df.iloc[:geneAmount, :], x="Gene", y=cmpName, color="k", ax=ax)

    ax.tick_params(axis="x", rotation=90)


