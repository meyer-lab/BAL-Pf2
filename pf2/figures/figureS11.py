"""Figure S10"""

import anndata
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotGeneral import rotate_xaxis, rotate_yaxis
from ..data_import import add_obs, combine_cell_types, condition_factors_meta_all
# from ..utilities import 
from scipy.stats import f_oneway, pearsonr

cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((10, 10), (1, 1))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    _, meta_df = condition_factors_meta_all(X)
    
    
    meta_df = meta_df.loc[meta_df["patient_category"] == "COVID-19"]
    # meta_df = meta_df.loc[meta_df["patient_category"] != "COVID-19"]
     
    pearson_df = pd.DataFrame(
        columns=correlates,
        index=correlates,
        dtype=float
    )
    
    for row in correlates:
        for column in correlates:
            two_meta_df = meta_df[[row, column]].dropna()
            result = pearsonr(
                two_meta_df.iloc[:, 0].values,
                two_meta_df.iloc[:, 1].values
            )
            pearson_df.loc[row, column] = result.pvalue

    mask = np.triu(np.ones_like(pearson_df, dtype=bool))
    for i in range(len(mask)):
        mask[i, i] = False
        
    
    sns.heatmap(
        data=pearson_df.to_numpy(),
        vmin=0,
        vmax=.05,
        xticklabels=correlates,
        yticklabels=correlates,
        mask=mask,
        cmap=cmap,
        cbar_kws={"label": "Pearson Correlation P-value"},
        ax=ax[0],
    )

    rotate_xaxis(ax[0], rotation=90)
    rotate_yaxis(ax[0], rotation=0)
        
        

        
    return f


correlates = [
    "age", "bmi", "cumulative_icu_days", "admit_sofa_score", "admit_aps_score",
    "cumulative_intubation_days", "BAL_amylase", "BAL_pct_neutrophils",
    "BAL_pct_macrophages", "BAL_pct_monocytes", "BAL_pct_lymphocytes",
    "BAL_pct_eosinophils", "BAL_pct_other",
    "temperature", "heart_rate", "systolic_blood_pressure",
    "diastolic_blood_pressure", "mean_arterial_pressure",
    "norepinephrine_rate", "respiratory_rate", "oxygen_saturation",
    "rass_score", "peep", "fio2", "plateau_pressure", "lung_compliance",
    "minute_ventilation", "abg_ph", "abg_paco2", "pao2fio2_ratio", "wbc_count",
    "bicarbonate", "creatinine", "albumin", "bilirubin", "crp", "d_dimer",
    "ferritin", "ldh", "lactic_acid", "procalcitonin", "nat_score",
    "steroid_dose", "episode_duration"
]