"""Figure A4: Prediction accuracy for no-penalty 2-pair LogReg and weights for important components for LogReg"""


import itertools
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from .common import subplotLabel, getSetup
from matplotlib.axes import Axes
import anndata
import pandas as pd
from sklearn.utils import resample
from pf2.data_import import add_obs, obs_per_condition
from ..tensor import correct_conditions
from tqdm import tqdm
from pf2.predict import predict_mortality
from pf2.figures.commonFuncs.plotGeneral import rotate_xaxis, rotate_yaxis


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((8, 12), (2, 1))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/andrew/bal_partial_fitted.h5ad")

    X = add_obs(X, "binary_outcome")
    labels = obs_per_condition(X, "binary_outcome")

    pair_logistic_regression(X, labels, ax[0])

    bootstrap_logistic_regression(X, labels, ax[1], trials=3)

    return f


def pair_logistic_regression(X: anndata.AnnData, labels, ax: Axes):
    """Plot factor weights for donor SLE prediction"""
    conditions_matrix = correct_conditions(X)
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


def bootstrap_logistic_regression(X: anndata.AnnData, labels, ax, trials: int = 5):
    """Bootstrap logistic regression"""
    conditions_matrix = correct_conditions(X)
    coefs = pd.DataFrame(
        index=np.arange(trials) + 1, columns=np.arange(conditions_matrix.shape[1])
    )

    all_pred_acc = []

    for trial in tqdm(range(trials)):
        boot_factors, boot_labels = resample(conditions_matrix, labels)
        pred_acc, coef = predict_mortality(boot_factors, boot_labels)
        coefs.iloc[trial, :] = coef
        all_pred_acc = np.append(all_pred_acc, pred_acc)

    ax.errorbar(
        np.arange(coefs.shape[1]) + 1,
        coefs.mean(axis=0),
        capsize=2,
        yerr=1.96 * coefs.std(axis=0) / np.sqrt(trials),
        linestyle="",
        marker=".",
        zorder=3,
    )

    ax.plot([0, 41], [0, 0], linestyle="--", color="k", zorder=0)
    ax.set_xticks(np.arange(conditions_matrix.shape[1]) + 1)
    ax.set_xticklabels(np.arange(conditions_matrix.shape[1]) + 1, fontsize=8)

    ax.set_xlim([0, conditions_matrix.shape[1] + 1])
    ax.grid(True)

    ax.set_ylabel("Logistic Regression Coefficient")
    ax.set_xlabel("PARAFAC2 Component")
    print("Average Prediction Accuracy: ", np.mean(all_pred_acc))
