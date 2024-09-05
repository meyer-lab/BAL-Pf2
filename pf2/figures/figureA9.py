"""
Figure A8:
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


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((6, 3), (1, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")

    meta = import_meta()
    conversions = convert_to_patients(X)

    patient_factor = pd.DataFrame(
        X.uns["Pf2_A"],
        index=conversions,
        columns=np.arange(X.uns["Pf2_A"].shape[1]) + 1,
    )
    meta = meta.loc[patient_factor.index, :]

    plsr_acc_df = plsr_acc_proba(patient_factor, meta)

    sns.barplot(data=plsr_acc_df, ax=ax[0])
    ax[0].set(ylim=[0, 1], ylabel="Accuracy")
    
    plot_plsr_auc_roc(patient_factor, meta, ax[1])
    
    return f
    
    
def plsr_acc_proba(patient_factor_matrix, meta_data):
    """Runs PLSR and obtains average prediction accuracy"""
    
    acc_df = pd.DataFrame(columns=["Overall", "C19", "nC19"])

    probabilities, labels = predict_mortality(
        patient_factor_matrix,
        meta_data,
        proba=True
    )

    probabilities = probabilities.round().astype(int)
    meta_data = meta_data.loc[~meta_data.index.duplicated()].loc[labels.index]

    covid_acc = accuracy_score(
        labels.loc[meta_data.loc[:, "patient_category"] == "COVID-19"],
        probabilities.loc[meta_data.loc[:, "patient_category"] == "COVID-19"]
    )
    nc_acc = accuracy_score(
        labels.loc[meta_data.loc[:, "patient_category"] != "COVID-19"],
        probabilities.loc[meta_data.loc[:, "patient_category"] != "COVID-19"]
    )
    acc = accuracy_score(labels, probabilities)

    acc_df.loc[
        0,
        :
        ] = [acc, covid_acc, nc_acc]

    return acc_df


def plot_plsr_auc_roc(patient_factor_matrix, meta_data, ax):
    """Runs PLSR and plots ROC AUC based on actual and prediction labels"""
    
    probabilities, labels = predict_mortality(
        patient_factor_matrix,
        meta_data,
        proba=True
    )

    probabilities = probabilities.round().astype(int)
    meta_data = meta_data.loc[~meta_data.index.duplicated()].loc[labels.index]

    RocCurveDisplay.from_predictions(labels.loc[meta_data.loc[:, "patient_category"] == "COVID-19"], 
                                     probabilities.loc[meta_data.loc[:, "patient_category"] == "COVID-19"],
                                     ax=ax, name="C19")
    RocCurveDisplay.from_predictions(labels.loc[meta_data.loc[:, "patient_category"] != "COVID-19"],
                                     probabilities.loc[meta_data.loc[:, "patient_category"] != "COVID-19"], 
                                     ax=ax, name="nC19")
    RocCurveDisplay.from_predictions(labels,
                                     probabilities, plot_chance_level=True,
                                     ax=ax, name="Overall")

