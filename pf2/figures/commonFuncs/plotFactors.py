from typing import Optional
from anndata import AnnData
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.cluster.hierarchy as sch
from matplotlib.patches import Patch
from matplotlib.axes import Axes

cmap = sns.diverging_palette(240, 10, as_cmap=True)


def plot_condition_factors(
    data: AnnData,
    ax: Axes,
    cond: str = "Condition",
    cond_group_labels: Optional[pd.Series] = None,
    color_key = None,
    group_cond = False,
):
    """Plots Pf2 condition factors"""
    pd.set_option("display.max_rows", None)
    yt = pd.Series(np.unique(data.obs[cond]))
    X = np.array(data.uns["Pf2_A"])

    XX = X
    X -= np.median(XX, axis=0)
    X /= np.std(XX, axis=0)

    ind = reorder_table(X)
    X = X[ind]
    yt = yt.iloc[ind]

    if cond_group_labels is not None:
        cond_group_labels = cond_group_labels.iloc[ind]
        if group_cond is True:
            ind = cond_group_labels.argsort()
            cond_group_labels = cond_group_labels.iloc[ind]
            X = X[ind]
            yt = yt.iloc[ind]
        ax.tick_params(axis="y", which="major", pad=20, length=0)
        if color_key is None:
            colors = sns.color_palette(
                n_colors=pd.Series(cond_group_labels).nunique()
            ).as_hex()
        else:
            colors = color_key
        lut = {}
        legend_elements = []
        for index, group in enumerate(pd.unique(cond_group_labels)):
            lut[group] = colors[index]
            legend_elements.append(Patch(color=colors[index], label=group))
        row_colors = pd.Series(cond_group_labels).map(lut)
        for iii, color in enumerate(row_colors):
            ax.add_patch(
                plt.Rectangle(
                    xy=(-0.05, iii),
                    width=0.05,
                    height=1,
                    color=color,
                    lw=0,
                    transform=ax.get_yaxis_transform(),
                    clip_on=False,
                )
            )
        ax.legend(handles=legend_elements, bbox_to_anchor=(0.18, 1.07))

    xticks = np.arange(1, X.shape[1] + 1)
    sns.heatmap(
        data=X,
        xticklabels=xticks,
        yticklabels=yt,
        ax=ax,
        center=0,
        cmap=cmap,
    )
    ax.tick_params(axis="y", rotation=0)
    ax.set(xlabel="Component")


def plot_eigenstate_factors(data: AnnData, ax: Axes):
    """Plots Pf2 eigenstate factors"""
    rank = data.uns["Pf2_B"].shape[1]
    xticks = np.arange(1, rank + 1)
    X = data.uns["Pf2_B"]
    X = X / np.max(np.abs(np.array(X)))
    yt = np.arange(1, rank + 1)

    sns.heatmap(
        data=X,
        xticklabels=xticks,
        yticklabels=yt,
        ax=ax,
        center=0,
        cmap=cmap,
        vmin=-1,
        vmax=1,
    )
    ax.set(xlabel="Component")


def plot_gene_factors(data: AnnData, ax: Axes, trim=True):
    """Plots Pf2 gene factors"""
    rank = data.varm["Pf2_C"].shape[1]
    X = np.array(data.varm["Pf2_C"])
    yt = data.var.index.values

    if trim is True:
        max_weight = np.max(np.abs(X), axis=1)
        kept_idxs = max_weight > 0.04
        X = X[kept_idxs]
        yt = yt[kept_idxs]

    ind = reorder_table(X)
    X = X[ind]
    X = X / np.max(np.abs(X))
    yt = [yt[ii] for ii in ind]
    xticks = np.arange(1, rank + 1)

    sns.heatmap(
        data=X,
        xticklabels=xticks,
        yticklabels=yt,
        ax=ax,
        center=0,
        cmap=cmap,
        vmin=-1,
        vmax=1,
    )
    ax.set(xlabel="Component")
    
    
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


def plot_factor_weight(X: AnnData, ax: Axes):
    """Plots weights from Pf2 model"""
    df = pd.DataFrame(data=np.transpose(X.uns["Pf2_weights"]), columns=["Value"])
    df["Value"] = df["Value"] / np.max(df["Value"])
    df["Component"] = np.arange(1, len(X.uns["Pf2_weights"]) + 1)
    sns.barplot(data=df, x="Component", y="Value", ax=ax)
    ax.tick_params(axis="x", rotation=90)


def reorder_table(projs: np.ndarray) -> np.ndarray:
    """Reorder a table's rows using heirarchical clustering"""
    assert projs.ndim == 2
    Z = sch.linkage(projs, method="complete", metric="cosine", optimal_ordering=True)
    return sch.leaves_list(Z)


