"""Figure Debug 1: Factor Matrices & Cell Embedding"""
import time

import datashader as ds
import datashader.transfer_functions as tf
from matplotlib import colormaps
from matplotlib.colors import to_hex
import pandas as pd
import seaborn as sns

from pf2.data_import import import_data
from pf2.figures.common import ds_show, get_canvas, getSetup, plot_gene_factors
from pf2.tensor import pf2

CONVERSIONS = {
    "CD8 T cells": "T Cells",
    "Monocytes1": "Monocytes",
    "Mac3 CXCL10": "Macrophages",
    "Monocytes2": "Monocytes",
    "B cells": "B Cells",
    "CD4 T cells": "T Cells",
    "CM CD8 T cells": "T Cells",
    "Tregs": "T-regulatory",
    "Plasma cells1": "B Cells",
    "Migratory DC CCR7": "Dendritic Cells",
    "Proliferating T cells": "Proliferating",
    "Monocytes3 HSPA6": "Monocytes",
    "Mac2 FABP4": "Macrophages",
    "DC2": "Dendritic Cells",
    "Mac4 SPP1": "Macrophages",
    "pDC": "Dendritic Cells",
    "Mac1 FABP4": "Macrophages",
    "Proliferating Macrophages": "Macrophages",
    "Mac6 FABP4": "Macrophages",
    "DC1 CLEC9A": "Dendritic Cells",
    "IFN resp. CD8 T cells": "T Cells",
    "NK/gdT cells": "NK Cells",
    "Mast cells": "Other",
    "Secretory cells": "Other",
    "Ciliated cells": "Other",
    "Epithelial cells": "Other",
    "Mac5 FABP4": "Macrophages",
    "Ionocytes": "Other",
}
SCATTER_COLORS = [to_hex(color) for color in colormaps["tab20"].colors]
COLORS = {
    "B Cells": "#ffbb78",
    "Dendritic Cells": "#98df8a",
    "Macrophages": "#ff7f0e",
    "Monocytes": "#aec7e8",
    "NK Cells": "#ff9896",
    "Other": "#9467bd",
    "Proliferating": "#d62728",
    "T Cells": "#1f77b4",
    "T-regulatory": "#2ca02c",
}


def makeFigure():
    start = time.time()
    data = import_data()
    print(data.shape)
    factors, _ = pf2(data, rank=40)
    print(f"Factorization Time: {time.time() - start}")

    axs, fig = getSetup((12, 4), (1, 4))

    # Patient factor

    ax = axs[0]

    sns.heatmap(
        factors.uns["Pf2_A"] / abs(factors.uns["Pf2_A"]).max(axis=0),
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        ax=ax,
        cbar=False,
    )
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("Patient Factor")

    # Gene factor

    ax = axs[1]

    plot_gene_factors(factors, ax, trim=True)
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_yticks([])
    ax.set_title("Gene Factor")

    # Cell State factor

    ax = axs[2]

    sns.heatmap(
        factors.uns["Pf2_B"] / abs(factors.uns["Pf2_B"]).max(axis=0),
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        ax=ax,
        cbar=False,
    )
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("Cell State Factor")

    # Cell Embedding

    ax = axs[3]

    embedding = pd.DataFrame(
        factors.obsm["embedding"], index=data.obs.index, columns=["x", "y"]
    )
    embedding.loc[:, "label"] = data.obs.loc[:, "cell_type"].values
    embedding.loc[:, "label"] = (
        embedding.loc[:, "label"].replace(CONVERSIONS).values
    )
    colors = COLORS.copy()

    canvas = get_canvas(embedding.values[:, :2].astype(float))
    result = tf.shade(
        agg=canvas.points(embedding, "x", "y", agg=ds.count_cat("label")),
        color_key=colors,
        how="eq_hist",
        alpha=255,
        min_alpha=255,
    )
    ds_show(result, ax)

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("PaCMAP 1")
    ax.set_ylabel("PaCMAP 2")
    ax.set_title("Cell PaCMAP")

    return fig
