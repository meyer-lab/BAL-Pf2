"""
Figure A10:
"""

import numpy as np
import pandas as pd
import anndata
import seaborn as sns
from ..data_import import convert_to_patients, import_meta
from ..predict import predict_mortality
from .common import subplotLabel, getSetup
import seaborn as sns

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((5, 4), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")

    meta = import_meta(drop_duplicates=False)
    conversions = convert_to_patients(X, sample=True)

    patient_factor = pd.DataFrame(
        X.uns["Pf2_A"],
        index=conversions,
        columns=np.arange(X.uns["Pf2_A"].shape[1]) + 1,
    )
    meta.set_index("sample_id", inplace=True)

    shared_indices = patient_factor.index.intersection(meta.index)
    patient_factor = patient_factor.loc[shared_indices, :]
    meta = meta.loc[shared_indices, :]

    labels, plsr_results_both = plsr_acc(patient_factor, meta)

    plot_plsr_loadings(plsr_results_both, ax[0], ax[1])
    ax[0].set(xlim=[-0.35, 0.35])
    ax[1].set(xlim=[-0.4, 0.4])

    plot_plsr_scores(plsr_results_both, meta, labels, ax[2], ax[3])
    ax[2].set(xlim=[-9.5, 9.5])
    ax[3].set(xlim=[-8.5, 8.5])

    return f


def plsr_acc(patient_factor_matrix, meta_data, n_components=1):
    """Runs PLSR and obtains average prediction accuracy"""

    _, labels, [c19_plsr, nc19_plsr] = predict_mortality(
        patient_factor_matrix, meta_data, n_components=n_components, proba=False
    )

    return labels, [c19_plsr, nc19_plsr]


def plot_plsr_loadings(plsr_results, ax1, ax2):
    """Runs PLSR and plots ROC AUC based on actual and prediction labels"""
    ax = [ax1, ax2]
    type_of_data = ["C19", "nC19"]

    for i in range(2):
        df_xload = pd.DataFrame(data=plsr_results[i].x_loadings_[:, 0], columns=["PLSR 1"])
        df_yload = pd.DataFrame(data=[[plsr_results[i].y_loadings_[0, 0]]], columns=["PLSR 1"])
        sns.swarmplot(
            data=df_xload,
            x="PLSR 1",
            ax=ax[i],
            color="k",
        )
        sns.swarmplot(
            data=df_yload,
            x="PLSR 1",
            ax=ax[i],
            color="r",
            
        )
        ax[i].set(xlabel="PLSR 1", ylabel="Pf2 Components", title=f"{type_of_data[i]}-loadings")


def plot_plsr_scores(plsr_results, meta_data, labels, ax1, ax2):
    """Runs PLSR and plots ROC AUC based on actual and prediction labels"""
    ax = [ax1, ax2]
    type_of_data = ["C19", "nC19"]

    meta_data = meta_data.loc[
        meta_data.loc[:, "patient_category"] != "Non-Pneumonia Control", :
    ]

    for i in range(2):
        if i == 0:
            score_labels = labels.loc[
                meta_data.loc[:, "patient_category"] == "COVID-19"
            ]
        else:
            score_labels = labels.loc[
                meta_data.loc[:, "patient_category"] != "COVID-19"
            ]

        pal = sns.color_palette()
        if i == 0: 
            numb1=0; numb2=2
        else:
            numb1=1; numb2=3
        
        df_xscores = pd.DataFrame(data=plsr_results[i].x_scores_[:, 0], columns=["PLSR 1"])
        sns.swarmplot(
            data=df_xscores,
            x="PLSR 1",
            ax=ax[i],
            hue=score_labels.to_numpy(),
            palette=[pal[numb1], pal[numb2]],
            hue_order=[1, 0],
        )
        ax[i].set(xlabel="PLSR 1", ylabel="Samples", title=f"{type_of_data[i]}-scores")