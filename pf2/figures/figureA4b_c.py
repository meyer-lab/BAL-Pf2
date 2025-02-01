"""
Figure A4b_c
"""

import numpy as np
import pandas as pd
import anndata
from sklearn.metrics import accuracy_score
import seaborn as sns
from ..data_import import convert_to_patients, import_meta
from ..predict import predict_mortality, predict_mortality_all
from .common import subplotLabel, getSetup
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import accuracy_score, roc_auc_score
from pf2.figures.commonFuncs.plotGeneral import bal_combine_bo_covid


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((6, 6), (2, 2))
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
    
    roc_auc = [False, True]
    for i in range(2):
        plsr_acc_df = pd.DataFrame([])
        for j in range(3):
            df = plsr_acc_proba(
                patient_factor, meta, n_components=j + 1, roc_auc=roc_auc[i]
            )
            df["Component"] = j + 1
            plsr_acc_df = pd.concat([plsr_acc_df, df], axis=0)

        plsr_acc_df = plsr_acc_df.melt(
            id_vars="Component", var_name="Category", value_name="Accuracy"
        )
        sns.barplot(
            data=plsr_acc_df, x="Component", y="Accuracy", hue="Category", ax=ax[i],
            hue_order=["C19", "nC19", "Overall"]
        )
        if roc_auc[i] is True:
            ax[i].set(ylim=[0, 1], ylabel="AUC ROC")
        else:
            ax[i].set(ylim=[0, 1], ylabel="Prediction Accuracy")

    for i in range(2):
        plot_plsr_auc_roc(patient_factor, meta, n_components=i + 1, ax=ax[i + 2])
        ax[i + 2].set(title=f"PLSR {i + 1} Components")

    return f


def plsr_acc_proba(patient_factor_matrix, meta_data, n_components=2, roc_auc=True):
    """Runs PLSR and obtains average prediction accuracy"""

    acc_df = pd.DataFrame(columns=["Overall", "C19", "nC19"])

    probabilities_all, labels_all = predict_mortality_all(
        patient_factor_matrix, n_components=n_components, meta=meta_data, proba=True
    )
    meta_data = meta_data.loc[~meta_data.index.duplicated()].loc[labels_all.index]

    if roc_auc:
        score = roc_auc_score
    else:
        score = accuracy_score
        
    covid_acc = score(
        labels_all.loc[meta_data.loc[:, "patient_category"] == "COVID-19"].to_numpy().astype(int),
        probabilities_all.loc[meta_data.loc[:, "patient_category"] == "COVID-19"].round().astype(int),
    )
    nc_acc = score(
        labels_all.loc[meta_data.loc[:, "patient_category"] != "COVID-19"].to_numpy().astype(int),
        probabilities_all.loc[meta_data.loc[:, "patient_category"] != "COVID-19"].round().astype(int),
    )
    acc = score(labels_all.to_numpy().astype(int), probabilities_all.round().astype(int))

    acc_df.loc[0, :] = [acc, covid_acc, nc_acc]

    return acc_df


def plot_plsr_auc_roc(patient_factor_matrix, meta_data, n_components, ax):
    """Runs PLSR and plots ROC AUC based on actual and prediction labels"""
    probabilities_all, labels_all = predict_mortality_all(
        patient_factor_matrix, n_components=n_components, meta=meta_data, proba=True)

    meta_data = meta_data.loc[~meta_data.index.duplicated()].loc[labels_all.index]

    RocCurveDisplay.from_predictions(
        labels_all.loc[meta_data.loc[:, "patient_category"] == "COVID-19"].to_numpy().astype(int),
        probabilities_all.loc[meta_data.loc[:, "patient_category"] == "COVID-19"],
        ax=ax,
        name="C19",
    )
    RocCurveDisplay.from_predictions(
        labels_all.loc[meta_data.loc[:, "patient_category"] != "COVID-19"].to_numpy().astype(int),
        probabilities_all.loc[meta_data.loc[:, "patient_category"] != "COVID-19"],
        ax=ax,
        name="nC19",
    )
    RocCurveDisplay.from_predictions(
        labels_all.to_numpy().astype(int), probabilities_all, plot_chance_level=True, ax=ax, name="Overall"
    )