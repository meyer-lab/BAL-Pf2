"""
Figure A9:
"""

import numpy as np
import pandas as pd
import anndata
from sklearn.metrics import accuracy_score
import seaborn as sns
from ..data_import import convert_to_patients, import_meta
from ..predict import predict_mortality
from .common import subplotLabel, getSetup
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import accuracy_score, roc_auc_score
from pf2.figures.commonFuncs.plotGeneral import bal_combine_bo_covid, rotate_xaxis
from ..data_import import add_obs, condition_factors_meta


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((6, 6), (3, 3))
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


    meta["TP"] = meta["icu_day"].transform(
        lambda x: "1TP" if x == 1 else ("2TP" if x == 2 else ("3TP" if x == 3 else ">=3TP"))
    )

    roc_auc = [False, True]
    axs = 0
    for i in range(2):
        for j, timepoint in enumerate(["1TP", "2TP", ">=3TP"]):
            plsr_acc_df = pd.DataFrame([])
            for k in range(3):
                meta_timepoint = meta.loc[meta["TP"] == timepoint, :]
                patient_factor_timepoint = patient_factor.loc[
                    meta["TP"] == timepoint, :
                ]
                df = plsr_acc_proba(
                    patient_factor_timepoint, meta_timepoint, n_components=k + 1, roc_auc=roc_auc[i]
                )
            
                df["Component"] = k + 1
                plsr_acc_df = pd.concat([plsr_acc_df, df], axis=0)
        
            plsr_acc_df = plsr_acc_df.melt(
                id_vars="Component", var_name="Category", value_name="Accuracy"
            )
            sns.barplot(
                data=plsr_acc_df, x="Component", y="Accuracy", hue="Category", ax=ax[axs]
            )
            ax[axs].set(title=timepoint)
            axs += 1
            if roc_auc[i] is True:
                ax[i].set(ylim=[0, 1], ylabel="AUC ROC")
            else:
                ax[i].set(ylim=[0, 1], ylabel="Prediction Accuracy")

        # plot_plsr_auc_roc(patient_factor, meta, ax[2])

    return f


def plsr_acc_proba(patient_factor_matrix, meta_data, n_components=2, roc_auc=True):
    """Runs PLSR and obtains average prediction accuracy"""

    acc_df = pd.DataFrame(columns=["Overall", "C19", "nC19"])

    probabilities, labels = predict_mortality(
        patient_factor_matrix, n_components=n_components, meta=meta_data, proba=True
    )

    probabilities = probabilities.round().astype(int)
    meta_data = meta_data.loc[~meta_data.index.duplicated()].loc[labels.index]

    if roc_auc:
        score = roc_auc_score
    else:
        score = accuracy_score

    covid_acc = score(
        labels.loc[meta_data.loc[:, "patient_category"] == "COVID-19"],
        probabilities.loc[meta_data.loc[:, "patient_category"] == "COVID-19"],
    )
    nc_acc = score(
        labels.loc[meta_data.loc[:, "patient_category"] != "COVID-19"],
        probabilities.loc[meta_data.loc[:, "patient_category"] != "COVID-19"],
    )
    acc = score(labels, probabilities)

    acc_df.loc[0, :] = [acc, covid_acc, nc_acc]

    return acc_df


def plot_plsr_auc_roc(patient_factor_matrix, meta_data, ax):
    """Runs PLSR and plots ROC AUC based on actual and prediction labels"""
    probabilities, labels = predict_mortality(
        patient_factor_matrix, meta_data, proba=True
    )
    meta_data = meta_data.loc[~meta_data.index.duplicated()].loc[labels.index]

    RocCurveDisplay.from_predictions(
        labels.loc[meta_data.loc[:, "patient_category"] == "COVID-19"],
        probabilities.loc[meta_data.loc[:, "patient_category"] == "COVID-19"],
        ax=ax,
        name="C19",
    )
    RocCurveDisplay.from_predictions(
        labels.loc[meta_data.loc[:, "patient_category"] != "COVID-19"],
        probabilities.loc[meta_data.loc[:, "patient_category"] != "COVID-19"],
        ax=ax,
        name="nC19",
    )
    RocCurveDisplay.from_predictions(
        labels, probabilities, plot_chance_level=True, ax=ax, name="Overall"
    )
