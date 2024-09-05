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
    ax, f = getSetup((8, 8), (2, 2))
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

    plsr_results_both = plsr_acc(patient_factor, meta)

    plot_plsr_loadings(plsr_results_both, ax[0], ax[1], text=False)
    ax[0].set(xlim=[-.4, .4], ylim=[-.4, .4])
    ax[1].set(xlim=[-.4, .4], ylim=[-.4, .4])
    
    # plot_plsr_loadings(plsr_results_both, ax[2], ax[3], text=True)
    
    return f
    
    
def plsr_acc(patient_factor_matrix, meta_data):
    """Runs PLSR and obtains average prediction accuracy"""

    acc, [c19_plsr, nc19_plsr] = predict_mortality(
        patient_factor_matrix,
        meta_data,
    )
    
    return [c19_plsr, nc19_plsr]


def plot_plsr_loadings(plsr_results, ax1, ax2, text=False):
    """Runs PLSR and plots ROC AUC based on actual and prediction labels"""
    

    ax = [ax1, ax2]
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
                c="k"
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
                        c="w"
                )
    
        ax[i].set(xlabel="PLSR 1", ylabel="PLSR 2", title = f"{type_of_data[i]}-loadings")


    





