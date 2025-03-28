"""
Figure A3b_c
"""

import pandas as pd
import anndata
import seaborn as sns
from ..data_import import condition_factors_meta, add_obs, combine_cell_types
from ..predict import predict_mortality_all, plsr_acc_proba
from .common import subplotLabel, getSetup
from sklearn.metrics import RocCurveDisplay


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    combine_cell_types(X)
    
    cond_fact_meta_df = condition_factors_meta(X)

    roc_auc = [False, True]
    for i in range(2):
        plsr_acc_df = pd.DataFrame([])
        for j in range(3):
            df = plsr_acc_proba(
                X, cond_fact_meta_df, n_components=j + 1, roc_auc=roc_auc[i]
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
        plot_plsr_auc_roc(X, cond_fact_meta_df, n_components=i + 1, ax=ax[i + 2])
        ax[i + 2].set(title=f"PLSR {i + 1} Components")

    return f



def plot_plsr_auc_roc(X, patient_factor_matrix, n_components, ax):
    """Runs PLSR and plots ROC AUC based on actual and prediction labels"""
    probabilities_all, labels_all = predict_mortality_all(X, 
        patient_factor_matrix, n_components=n_components, proba=True)

    RocCurveDisplay.from_predictions(
        labels_all.loc[patient_factor_matrix.loc[:, "patient_category"] == "COVID-19"].to_numpy().astype(int),
        probabilities_all.loc[patient_factor_matrix.loc[:, "patient_category"] == "COVID-19"],
        ax=ax,
        name="C19",
    )
    RocCurveDisplay.from_predictions(
        labels_all.loc[patient_factor_matrix.loc[:, "patient_category"] != "COVID-19"].to_numpy().astype(int),
        probabilities_all.loc[patient_factor_matrix.loc[:, "patient_category"] != "COVID-19"],
        ax=ax,
        name="nC19",
    )
    RocCurveDisplay.from_predictions(
        labels_all.to_numpy().astype(int), probabilities_all, plot_chance_level=True, ax=ax, name="Overall"
    )