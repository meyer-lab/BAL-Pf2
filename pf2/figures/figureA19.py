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
from pf2.figures.figureA9 import plsr_acc_proba, plot_plsr_auc_roc

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
    lambda x: "1-2TP" if x in [1, 2] else ("3-8TP" if x in [3, 4, 5, 6, 7, 8]  else ">=9TP"))

    # meta = meta.set_index("sample_id")
    roc_auc = [False, True]
    axs = 0
    for i in range(2):
        for j, timepoint in enumerate(["1-2TP", "3-8TP", ">=9TP"]):
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
                data=plsr_acc_df, x="Component", y="Accuracy", hue="Category", ax=ax[axs], hue_order=["C19", "nC19", "Overall"]
            )
            ax[axs].set(title=timepoint)
            if roc_auc[i] is True:
                ax[axs].set(ylim=[0, 1], ylabel="AUC ROC")
            else:
                ax[axs].set(ylim=[0, 1], ylabel="Prediction Accuracy")
                
            if i == 0:
                plot_plsr_auc_roc(patient_factor_timepoint, meta_timepoint, ax[axs+6])
                ax[axs+6].set(title=timepoint)
            
            axs += 1
    
          

    return f
