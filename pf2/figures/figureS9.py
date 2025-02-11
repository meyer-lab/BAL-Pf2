"""
Figure S9
"""

import pandas as pd
import anndata
import numpy as np
import seaborn as sns
from ..data_import import condition_factors_meta
from ..predict import plsr_acc
from .common import subplotLabel, getSetup

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((20, 4), (1, 1))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    
    cond_fact_meta_df = condition_factors_meta(X)
    trials = 50
    bootstrapp_plsr_df = pd.DataFrame([])
    
    for trial in range(trials):
        boot_index = np.random.choice(
            cond_fact_meta_df.shape[0], replace=True, size=cond_fact_meta_df.shape[0]
        )
        boot_cond_fact_meta_df = cond_fact_meta_df.iloc[boot_index, :]
        boot_cond_fact_meta_df.index = [
            f"{idx}_{i}" if boot_index.tolist().count(idx) > 1 else idx
            for i, idx in enumerate(boot_index)
        ]
        _, plsr_results_both = plsr_acc(X, boot_cond_fact_meta_df, n_components=1)
    
        type_of_data = ["C19", "nC19"]
        
        for i in range(2):
            x_load = np.abs(plsr_results_both[i].x_loadings_[:, 0])
            df_xload = pd.DataFrame(data=x_load, columns=["PLSR 1"])
            df_xload["Component"] = np.arange(df_xload.shape[0]) + 1
            df_xload["Trial"] = trial
            df_xload["Status"] = type_of_data[i]
        
            bootstrapp_plsr_df = pd.concat([bootstrapp_plsr_df, df_xload], axis=0)
            
            
    sns.barplot(bootstrapp_plsr_df, x="Component", y="PLSR 1", hue="Status", ax=ax[0])


        
    return f


