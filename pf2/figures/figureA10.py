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
from ..figures.commonFuncs.plotGeneral import bal_combine_bo_covid


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((8, 4), (1, 2))
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

    # labels, plsr_results_both = plsr_acc(patient_factor, meta)
    labels, plsr_overall = plsr_acc_overall(patient_factor, meta)
    # plot_plsr_loadings_overall(plsr_overall, ax[0], text=False)

    plot_plsr_scores_overall(plsr_overall, meta, labels, ax[0])
    
    
    
    # plot_plsr_loadings_overall(plsr_overall, ax[1], text=True)

    # plot_plsr_loadings(plsr_results_both, ax[0], ax[1], text=False)
    # ax[0].set(xlim=[-.4, .4], ylim=[-.4, .4])
    # ax[1].set(xlim=[-.4, .4], ylim=[-.4, .4])
    
    # plot_plsr_loadings(plsr_results_both, ax[0], ax[1], text=True)
    # plot_plsr_scores(plsr_results_both, meta, labels, ax[2], ax[3])
    # ax[2].set(xlim=[-9, 9], ylim=[-9, 9])
    # ax[3].set(xlim=[-8, 8], ylim=[-8, 8])
    
    return f
    
    
def plsr_acc(patient_factor_matrix, meta_data):
    """Runs PLSR and obtains average prediction accuracy"""

    labels, [c19_plsr, nc19_plsr] = predict_mortality(
        patient_factor_matrix,
        meta_data,
        proba=False
    )
    
    return labels, [c19_plsr, nc19_plsr]

def plsr_acc_overall(patient_factor_matrix, meta_data):
    """Runs PLSR and obtains average prediction accuracy"""

    predictions_overall, overall_plsr = predict_mortality(
        patient_factor_matrix,
        meta_data,
        proba=False,
        overall=True
    )
    
    return predictions_overall, overall_plsr


def plot_plsr_loadings(plsr_results, ax1, text=False):
    """Runs PLSR and plots ROC AUC based on actual and prediction labels"""
    ax = [ax1]
    type_of_data = ["C19", "nC19"]

    for i in range(2):
        ax[i].scatter(
            np.abs(plsr_results[i].y_loadings_[0, 0]),
            np.abs(plsr_results[i].y_loadings_[0, 1]),
            c="tab:red"
        )
        ax[i].scatter(
                plsr_results[i].x_loadings_[:, 0],
                plsr_results[i].x_loadings_[:, 1],
                # c="k"
            )
        if text:
            for index, component in enumerate(plsr_results[i].coef_.index):
                    ax[i].text(
                        plsr_results[i].x_loadings_[index, 0],
                        plsr_results[i].x_loadings_[index, 1] - 0.001,
                        ha="center",
                        ma="center",
                        va="center",
                        s=component,
                        # c="w"
                )
    
        ax[i].set(xlabel="PLSR 1", ylabel="PLSR 2", title = f"{type_of_data[i]}-loadings")
        
def plot_plsr_loadings_overall(plsr_results, ax1, text=False):
    """Runs PLSR and plots ROC AUC based on actual and prediction labels"""
    ax = ax1



    ax.scatter(
        np.abs(plsr_results.y_loadings_[0, 0]),
        np.abs(plsr_results.y_loadings_[0, 1]),
        c="tab:red"
    )
    ax.scatter(
            plsr_results.x_loadings_[:, 0],
            plsr_results.x_loadings_[:, 1],
            # c="k"
        )
    if text:
        for index, component in enumerate(plsr_results.coef_.index):
                ax.text(
                    plsr_results.x_loadings_[index, 0],
                    plsr_results.x_loadings_[index, 1] - 0.001,
                    ha="center",
                    ma="center",
                    va="center",
                    s=component,
                    # c="w"
            )

    ax.set(xlabel="PLSR 1", ylabel="PLSR 2", title = f"loadings")
    
        
def plot_plsr_scores(plsr_results, meta_data, labels, ax1, ax2):
    """Runs PLSR and plots ROC AUC based on actual and prediction labels"""
    ax = [ax1, ax2]
    type_of_data = ["C19", "nC19"]
    
    meta_data = meta_data.loc[
        meta_data.loc[:, "patient_category"] != "Non-Pneumonia Control",
        :
    ]
    
    for i in range(2):
        if i == 0: 
            score_labels = labels.loc[meta_data.loc[:, "patient_category"] == "COVID-19"]
        else: 
            score_labels = labels.loc[meta_data.loc[:, "patient_category"] != "COVID-19"]
            
        sns.scatterplot(
                x=plsr_results[i].x_scores_[:, 0],
                y=plsr_results[i].x_scores_[:, 1],
                hue=score_labels.to_numpy(),
                ax=ax[i]
            )
    
        ax[i].set(xlabel="PLSR 1", ylabel="PLSR 2", title = f"{type_of_data[i]}-loadings")
        
def plot_plsr_scores_overall(plsr_results, meta_data, labels, ax1):
    """Runs PLSR and plots ROC AUC based on actual and prediction labels"""

    meta_data = meta_data.loc[
        meta_data.loc[:, "patient_category"] != "Non-Pneumonia Control",
        :
    ]
    
    meta_data = bal_combine_bo_covid(meta_data)
    print(len(plsr_results.x_scores_[:, 0]))
    print(len(meta_data["Status"].to_numpy()))
    
    sns.scatterplot(
            x=plsr_results.x_scores_[:, 0],
            y=plsr_results.x_scores_[:, 1],
            hue=meta_data["Status"].to_numpy(),
            ax=ax1
        )

    ax1.set(xlabel="PLSR 1", ylabel="PLSR 2", title = f"loadings")




