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
from ..utilities import cell_count_perc_df
from scipy.stats import f_oneway, pearsonr
from .figureS1 import move_index_to_column, aggregate_anndata
cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, reverse=True, as_cmap=True)


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((10, 3), (1, 1))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    combine_cell_types(X)
    
    cell_comp_df = cell_count_perc_df(X, celltype="combined_cell_type")
    # print(cell_comp_df)
    _, meta_df = condition_factors_meta_all(X)
    cell_comp_df = cell_comp_df.pivot(
        index=["sample_id"],
        columns="Cell Type",
        values="Cell Type Percentage",
    )
    
    cell_comp_df = cell_comp_df.fillna(0)
    # cell_comp_df = move_index_to_column(cell_comp_df)
    print(cell_comp_df)
    # _, meta_df = condition_factors_meta_all(X)
    
    # print(cell_comp_df[meta_df.index, :])
    # a
    # meta_df = meta_df.loc[meta_df["patient_category"] == "COVID-19"]
    # meta_df = meta_df.loc[meta_df["patient_category"] != "COVID-19"]
     
    pearson_df = pd.DataFrame(
        columns=correlates,
        index=cell_comp_df.columns,
        dtype=float
    )
    
    for row in cell_comp_df.columns:
        for column in correlates:
            one_meta_df = meta_df[column].dropna()
            common = one_meta_df.index.intersection(cell_comp_df.index)
            one_meta_df = one_meta_df.loc[common]
            cell_comp_partial_df = cell_comp_df.loc[common, :]
            # print(len(one_meta_df.values))
            # print(len(cell_comp_df.loc[:, row].values))
            if len(one_meta_df.values)>2:
                result = pearsonr(
                    one_meta_df.values,
                    cell_comp_partial_df.loc[:, row].values
                )
                print(result.pvalue)
                pearson_df.loc[row, column] = result.pvalue

    print(pearson_df)   
    # mask = np.triu(np.ones_like(pearson_df, dtype=bool))
    # for i in range(len(mask)):
    #     mask[i, i] = False
        
    
    sns.heatmap(
        data=pearson_df.to_numpy(),
        vmin=0,
        vmax=.05,
        xticklabels=correlates,
        yticklabels=cell_comp_df.columns,
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