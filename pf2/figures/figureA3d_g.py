"""
Figure A3d_g
"""

import numpy as np
import pandas as pd
import anndata
import seaborn as sns
from ..data_import import condition_factors_meta
from ..predict import plsr_acc
from .common import subplotLabel, getSetup
import seaborn as sns

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((3, 7), (4, 1))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")

    cond_fact_meta_df = condition_factors_meta(X)
    
    labels, plsr_results_both = plsr_acc(X, cond_fact_meta_df, n_components=1)
    
    
    overlapping_idx = cond_fact_meta_df.index.intersection(labels.index)
    cond_fact_meta_df_filtered = cond_fact_meta_df.loc[overlapping_idx]
    cond_fact_meta_df_filtered = cond_fact_meta_df_filtered[cond_fact_meta_df_filtered.loc[:, "patient_category"] != "COVID-19"]
    cond_fact_meta_df_filtered = cond_fact_meta_df_filtered["patient_category"]
    
    
    print(cond_fact_meta_df_filtered)
    
    plot_plsr_scores_extra(plsr_results_both, cond_fact_meta_df, cond_fact_meta_df_filtered, ax[0])
    

    
    # plot_plsr_loadings(plsr_results_both, ax[0], ax[1])
    # ax[0].set(xlim=[-0.35, 0.35])
    # ax[1].set(xlim=[-0.35, 0.35])

    # plot_plsr_scores(plsr_results_both, cond_fact_meta_df, labels, ax[2], ax[3])
    # ax[2].set(xlim=[-7, 7])
    # ax[3].set(xlim=[-9.5, 9.5])

    return f


def plot_plsr_loadings(plsr_results, ax1, ax2):
    """Runs PLSR and plots ROC AUC based on actual and prediction labels"""
    ax = [ax1, ax2]
    type_of_data = ["C19", "nC19"]

    for i in range(2):
        x_load = plsr_results[i].x_loadings_[:, 0]
        if i == 1:
            x_load =-1*x_load
        df_xload = pd.DataFrame(data=x_load, columns=["PLSR 1"])
        df_xload["Component"] = np.arange(df_xload.shape[0]) + 1
        print(df_xload.sort_values(by="PLSR 1"))
        y_load = plsr_results[i].y_loadings_[0, 0]
        if i == 1:
            y_load =-1*y_load
        df_yload = pd.DataFrame(data=[[y_load]], columns=["PLSR 1"])
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


def plot_plsr_scores(plsr_results, cond_fact_meta_df, labels, ax1, ax2):
    """Runs PLSR and plots ROC AUC based on actual and prediction labels"""
    ax = [ax1, ax2]
    type_of_data = ["C19", "nC19"]

    cond_fact_meta_df = cond_fact_meta_df.loc[
        cond_fact_meta_df.loc[:, "patient_category"] != "Non-Pneumonia Control", :
    ]

    for i in range(2):
        if i == 0:
            score_labels = labels.loc[
                cond_fact_meta_df.loc[:, "patient_category"] == "COVID-19"
            ]
        else:
            score_labels = labels.loc[
                cond_fact_meta_df.loc[:, "patient_category"] != "COVID-19"
            ]

        pal = sns.color_palette()
        if i == 0: 
            numb1=3; numb2=2
        else:
            numb1=1; numb2=0
        
        x_scores = plsr_results[i].x_scores_[:, 0]
        if i == 1:
            x_scores =-1*x_scores
        df_xscores = pd.DataFrame(data=x_scores, columns=["PLSR 1"])
        sns.swarmplot(
            data=df_xscores,
            x="PLSR 1",
            ax=ax[i],
            hue=score_labels.to_numpy(),
            palette=[pal[numb1], pal[numb2]],
            hue_order=[1, 0],
        )
        ax[i].set(xlabel="PLSR 1", ylabel="Samples", title=f"{type_of_data[i]}-scores")
        
        
def plot_plsr_scores_extra(plsr_results, cond_fact_meta_df, labels, ax):
    """Runs PLSR and plots ROC AUC based on actual and prediction labels"""
    type_of_data = ["nC19"]
    score_labels = labels.loc[
                cond_fact_meta_df.loc[:, "patient_category"] != "COVID-19"
            ]
    x_scores = plsr_results[1].x_scores_[:, 0]
    df_xscores = pd.DataFrame(data=x_scores, columns=["PLSR 1"])
    sns.swarmplot(
        data=df_xscores,
        x="PLSR 1",
        ax=ax,
        hue=score_labels.to_numpy(),
    )
    ax.set(xlabel="PLSR 1", ylabel="Samples", title=f"{type_of_data}-scores")