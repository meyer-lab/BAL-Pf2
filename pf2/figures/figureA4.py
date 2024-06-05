"""
Lupus: Prediction accuracy for all two
pair logistic regression combinations
"""

from anndata import read_h5ad
import itertools
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from .common import subplotLabel, getSetup
from matplotlib.axes import Axes
import anndata
import pandas as pd
from pf2.data_import import convert_to_patients, import_meta
from ..tensor import correct_conditions


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((7, 6), (1, 1))

    X = read_h5ad("balPf240comps_factors.h5ad")

    meta = import_meta()
    meta = meta.loc[~meta.loc[:, "patient_id"].duplicated(), :]
    meta = meta.set_index("patient_id", drop=True)

    conversions = convert_to_patients(X)
    patient_factor = pd.DataFrame(
        X.uns["Pf2_A"],
        index=conversions,
        columns=np.arange(X.uns["Pf2_A"].shape[1]) + 1,
    )

    patient_factor = patient_factor.loc[
        patient_factor.index.isin(meta.index), :
    ]
 
    labels = patient_factor.index.to_series().replace(
        meta.loc[:, "binary_outcome"]
    )

    # pair_logistic_regression(correct_conditions(X), labels, ax[0])
    pair_logistic_regression(patient_factor.to_numpy(), labels, ax[0])

    return f


def pair_logistic_regression(conditions_matrix, labels, ax: Axes):
    """Plot factor weights for donor SLE prediction"""
    lrmodel = LogisticRegression(penalty=None)
   
    all_comps = np.arange(conditions_matrix.shape[1])
    acc = np.zeros((conditions_matrix.shape[1], conditions_matrix.shape[1]))

    for comps in itertools.product(all_comps, all_comps):
        if comps[0] >= comps[1]:
            compFacs = conditions_matrix[:, [comps[0], comps[1]]]
            LR_CoH = lrmodel.fit(compFacs, labels)
            acc[comps[0], comps[1]] = LR_CoH.score(compFacs, labels)
            acc[comps[1], comps[0]] = acc[comps[0], comps[1]]

    mask = np.triu(np.ones_like(acc, dtype=bool))

    for i in range(len(mask)):
        mask[i, i] = False

    sns.heatmap(
        data=acc,
        vmin=0.5,
        vmax=0.8,
        xticklabels=all_comps + 1,
        yticklabels=all_comps + 1,
        mask=mask,
        cbar_kws={"label": "Prediction Accuracy"},
        ax=ax,
    )

    ax.set(xlabel="Component", ylabel="Component")
    rotate_xaxis(ax, rotation=0)
    rotate_yaxis(ax, rotation=0)
    
    
def rotate_xaxis(ax, rotation=90):
    """Rotates text by 90 degrees for x-axis"""
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=rotation)


def rotate_yaxis(ax, rotation=90):
    """Rotates text by 90 degrees for y-axis"""
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(labels=ax.get_yticklabels(), rotation=rotation)