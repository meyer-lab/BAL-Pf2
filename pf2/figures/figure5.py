"""Figure 5: PacMAP plots"""
import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import read_h5ad
from matplotlib.lines import Line2D
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from tqdm import tqdm

from pf2.data_import import convert_to_patients, import_meta
from pf2.figures.common import (
    DEFAULT_CMAP,
    DIVERGING_CMAP,
    ds_show,
    get_canvas,
    getSetup,
)
from pf2.predict import predict_mortality

META_COLS = {
    "binary_outcome": "categorical",
    "age": "numeric",
    "episode_etiology": "categorical",
    "immunocompromised_flag": "categorical",
    "BAL_pct_neutrophils": "numeric",
    "BAL_pct_macrophages": "numeric",
}
SCATTER_COLORS = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]


def makeFigure():
    meta = import_meta()
    data = read_h5ad("factor_cache/factors.h5ad", backed="r")

    meta = meta.set_index("patient_id", drop=True)
    meta = meta.loc[~meta.index.duplicated()]

    names = ["Cells", "Genes", "Patients"]
    embeddings = [
        data.obsm["embedding"],
        data.varm["embedding"],
        data.uns["embedding"],
    ]
    for index, name in enumerate(names):
        embedding = embeddings[index]
        if name == "Patients":
            patient_map = (
                data.obs.loc[:, ["condition_unique_idxs", "patient_id"]]
                .set_index("condition_unique_idxs", drop=True)
                .squeeze()
            )
            patient_map = patient_map.loc[~patient_map.index.duplicated()]
            patient_map = patient_map.sort_index(ascending=True)
            embedding = pd.DataFrame(
                embedding, index=patient_map.values, columns=["x", "y"]
            )
            meta = meta.loc[patient_map.values, :]
        else:
            embedding = pd.DataFrame(embedding, columns=["x", "y"])

        embeddings[index] = embedding

    axs, fig = getSetup((6, 6), (3, 3))

    # Plot factors
    for name, embedding, ax in zip(names, embeddings, axs[:3]):
        canvas = get_canvas(embedding.values)
        result = tf.shade(
            agg=canvas.points(embedding, "x", "y", agg=ds.count()),
            cmap=DEFAULT_CMAP,
            span=(-1, 1),
            how="linear",
            min_alpha=255,
        )
        ds_show(result, ax)

        ax.set_xlabel("PaCMAP 1")
        ax.set_ylabel("PaCMAP 2")
        ax.set_title(name)

    # Patient variation
    embedding = embeddings[2]
    for (key, value), ax in zip(META_COLS.items(), axs[3:]):
        canvas = get_canvas(embedding.values[:, :2])
        if value == "categorical":
            le = LabelEncoder()

            categories = le.fit_transform(meta.loc[:, key])
            embedding.loc[:, "label"] = categories
            embedding["label"] = embedding.loc[:, "label"].astype(
                "category"
            )
            colors = {
                le.classes_[i]: SCATTER_COLORS[i]
                for i in range(len(le.classes_))
            }
            result = tf.shade(
                agg=canvas.points(
                    embedding, "x", "y", agg=ds.count_cat("label")
                ),
                color_key=SCATTER_COLORS,
                how="eq_hist",
                alpha=255,
                min_alpha=255,
            )
            ds_show(result, ax)
            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color=color,
                    label=name,
                    markerfacecolor=color,
                    markersize=8,
                )
                for name, color in colors.items()
            ]
            ax.legend(handles=legend_elements)
        elif value == "numeric":
            embedding.loc[:, "label"] = meta.loc[:, key]
            span = (
                embedding.loc[:, "label"].min(),
                embedding.loc[:, "label"].max(),
            )
            psm = plt.pcolormesh([span, span], cmap=DIVERGING_CMAP)
            result = tf.shade(
                agg=canvas.points(embedding, "x", "y", agg=ds.mean("label")),
                cmap=DIVERGING_CMAP,
                span=span,
                how="linear",
                alpha=255,
                min_alpha=255,
            )
            ds_show(result, ax)
            plt.colorbar(psm, ax=ax)

        ax.set_xlabel("PaCMAP 1")
        ax.set_ylabel("PaCMAP 2")
        ax.set_title(key)

    return fig
