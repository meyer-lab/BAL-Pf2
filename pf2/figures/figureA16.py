"""
Figure A16:
"""

from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
import seaborn as sns
from pf2.figures.commonFuncs.plotGeneral import bal_combine_bo_covid
from ..data_import import add_obs, condition_factors_meta



def makeFigure():
    ax, f = getSetup((8, 8), (2, 2))
    subplotLabel(ax)

    X = read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    
    cond_fact_meta_df = condition_factors_meta(X)
    cond_fact_meta_df = bal_combine_bo_covid(cond_fact_meta_df)
    pat_df = cond_fact_meta_df[["patient_id", "icu_day", "Status"]]
    sns.violinplot(data=pat_df, x="Status", y="icu_day", hue="Status", cut=0, ax=ax[0])
    
    
    pat_df.loc[pat_df["icu_day" == 1], "Day"] = "Day1"
    pat_df.loc[pat_df["icu_day" != 1], "Day"] = "Day>1"
    
                
    count_df = (
            pat_df.groupby(["Status", "Day"], observed=True).size().reset_index(name="Sample Count")
        )
    total = count_df["Sample Count"].sum()
    count_df["Sample Count"] = count_df["Sample Count"] / total
    
    sns.barplot(data=count_df, x="Day", y="Sample Count", hue="Status", ax=ax[1])

    
    for i in range(2):
        if i == 0:
            day_df = pat_df[pat_df["icu_day"] == 1]
            day = "Day 1"
        else:
            day_df = pat_df[pat_df["icu_day"] != 1]
            day = "Day >1"
              
        count_df = (
            day_df.groupby(["Status"], observed=True).size().reset_index(name="Sample Count")
        )
        total = count_df["Sample Count"].sum()
        count_df["Sample Count"] = count_df["Sample Count"] / total
        sns.barplot(data=count_df, x="Status", y="Sample Count", hue="Status", ax=ax[i+2])
        ax[i+2].set_title(day)


    return f
