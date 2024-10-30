"""
Figure A16:
"""

from anndata import read_h5ad
from .common import (
    subplotLabel,
    getSetup,
)
import seaborn as sns
from pf2.figures.commonFuncs.plotGeneral import bal_combine_bo_covid, rotate_xaxis
from ..data_import import add_obs, condition_factors_meta
import pandas as pd
import numpy as np
from ..data_import import convert_to_patients, import_meta


def makeFigure():
    ax, f = getSetup((8, 8), (2, 2))
    subplotLabel(ax)

    X = read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    
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
    
    meta = bal_combine_bo_covid(meta)
    pat_df = meta[["patient_id", "icu_day", "Status"]]
    order = np.unique(pat_df["Status"])
    
    sns.violinplot(data=pat_df, x="Status", y="icu_day", hue="Status", hue_order=order, order=order, cut=0, ax=ax[0])
    
    print(pat_df)
    pat_df["Day"] = pat_df.groupby("patient_id")["icu_day"].transform(
        lambda x: "1TP" if x.nunique() == 1 else ("2TP" if x.nunique() == 2 else ">=3TP")
    )
    print(pat_df.loc[pat_df["patient_id"] == 476])
      
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
    ax[2].set(ylabel="Overall Patient Proportion")

    for i in range(3):
        rotate_xaxis(ax[i])


    return f
