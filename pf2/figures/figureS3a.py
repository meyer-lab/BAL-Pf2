"""
Figure S3: PLSR scores for patients with non-COVID-19 conditions
"""

import pandas as pd
import anndata
import seaborn as sns
from ..data_import import condition_factors_meta
from ..predict import plsr_acc
from .common import subplotLabel, getSetup

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((3, 2), (1, 1))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")

    cond_fact_meta_df = condition_factors_meta(X)
    
    labels, plsr_results_both = plsr_acc(X, cond_fact_meta_df, n_components=1)
    
    overlapping_idx = cond_fact_meta_df.index.intersection(labels.index)
    cond_fact_meta_df_filtered = cond_fact_meta_df.loc[overlapping_idx]
    cond_fact_meta_df_filtered = cond_fact_meta_df_filtered[cond_fact_meta_df_filtered.loc[:, "patient_category"] != "COVID-19"]
    cond_fact_meta_df_filtered = cond_fact_meta_df_filtered["patient_category"]

    plot_plsr_scores_extra(plsr_results_both, cond_fact_meta_df, cond_fact_meta_df_filtered, ax[0])
    ax[0].set(xlim=[-9.5, 9.5])

    return f
        
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