from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
from pf2.data_import import import_data, import_meta
import numpy as np
import seaborn as sns
import pandas as pd
from .commonFuncs.plotFactors import bot_top_genes
from .commonFuncs.plotGeneral import plot_avegene_per_status
from matplotlib.axes import Axes
import anndata


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((20, 15), (4, 3))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("balPf240comps_factors.h5ad")
    patient_id_X = np.unique(X.obs["patient_id"])
    meta = import_meta()
    reduced_meta = meta.loc[meta["patient_id"].isin(patient_id_X)][["patient_id", "binary_outcome"]].drop_duplicates()
    
    binary_outcome = np.empty(X.shape[0])
    for i, patient in enumerate(X.obs["patient_id"]):
        binary_outcome[i] = reduced_meta.loc[reduced_meta["patient_id"] == patient]["binary_outcome"].to_numpy()
        
    X.obs["binary_outcome"] = binary_outcome

    genes = bot_top_genes(X, cmp=1, geneAmount=6)

    for i, gene in enumerate(np.ravel(genes)):
        plot_avegene_per_status(X, gene, ax[i])
        rotate_xaxis(ax[i])

    return f

    
def rotate_xaxis(ax, rotation=90):
    """Rotates text by 90 degrees for x-axis"""
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=rotation)