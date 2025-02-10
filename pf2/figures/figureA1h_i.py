"""
Figure Ah_i:
"""

from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
import seaborn as sns
from pf2.figures.commonFuncs.plotGeneral import rotate_xaxis
from ..data_import import add_obs, condition_factors_meta
import numpy as np


def makeFigure():
    ax, f = getSetup((8, 8), (2, 2))
    subplotLabel(ax)

    X = read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    
    cond_fact_meta_df = condition_factors_meta(X)
    cond_fact_meta_df = cond_fact_meta_df[cond_fact_meta_df["patient_category"] != "Non-Pneumonia Control"]

    pat_df = cond_fact_meta_df[["patient_id", "ICU Day", "Status"]]
    order = np.unique(pat_df["Status"])
    
    sns.violinplot(data=pat_df, x="Status", y="ICU Day", hue="Status", hue_order=order, order=order, cut=0, ax=ax[0])
    
    pat_df["Day"] = pat_df.groupby("patient_id")["ICU Day"].transform(
        lambda x: "1TP" if x.nunique() == 1 else ("2TP" if x.nunique() == 2 else ">=3TP")
    )
      
    count_df = (
            pat_df.groupby(["Status", "Day"], observed=True).size().reset_index(name="Sample Count")
        )
    total = count_df["Sample Count"].sum()
    count_df["Sample Count"] = count_df["Sample Count"] / total
    
    sns.barplot(data=count_df, x="Day", y="Sample Count", hue="Status",ax=ax[1])
    ax[1].set(ylabel="Sample Proportion")
    
    pat_tp_df = pat_df.loc[pat_df["Day"] == "1TP"]
    pat_tp_count_df = (
        pat_tp_df.groupby(["Status"], observed=True).size().reset_index(name="Sample Count")
    )
    total = pat_tp_count_df["Sample Count"].sum()
    pat_tp_count_df["Sample Count"] = pat_tp_count_df["Sample Count"] / total

    sns.barplot(data=pat_tp_count_df, x="Status", y="Sample Count", hue="Status", ax=ax[2])
    ax[2].set(ylabel="Overall Patient Proportion: 1TP")

    for i in range(3):
        rotate_xaxis(ax[i])


    return f
