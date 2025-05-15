"""Table S1"""

import anndata
from .common import getSetup
from ..data_import import meta_raw_df, bal_combine_bo_covid
from ..correlation import meta_groupings
import pandas as pd
import numpy as np

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((15, 15), (5, 5))

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")

    all_meta_df = meta_raw_df(X, all=True)
    all_meta_df = bal_combine_bo_covid(all_meta_df)
    # all_meta_df = all_meta_df[all_meta_df["patient_category"] != "Non-Pneumonia Control"]

    combined_df = pd.DataFrame([])
    for i, corr in enumerate(meta_groupings):
        counts = all_meta_df.groupby([corr, "Status"]).size().reset_index(name='Count')
        total_counts = counts.groupby(corr)['Count'].transform('sum')
        counts['Percentage'] = counts['Count'] / total_counts * 100
        counts['Formatted'] = counts.apply(lambda row: f"{row['Percentage']:.1f}% ({row['Count']})", axis=1)
        
        pivot_table = counts.pivot(index=corr, columns='Status', values='Formatted')
        pivot_table['Category'] = corr + f" ({np.sum(total_counts.unique())})"
        pivot_table.index = [f"{idx} ({total_counts[counts[corr] == idx].iloc[0]})" for idx in pivot_table.index]
        
        combined_df = pd.concat([combined_df, pivot_table], axis=0)

    combined_df = combined_df.reset_index().rename(columns={'index': 'Category_Index'})
    combined_df = combined_df[['Category', 'Category_Index'] + [col for col in combined_df.columns if col not in ['Category', 'Category_Index']]]
    print(combined_df)
    
    combined_df.to_csv("pf2/data/meta_groupings_summary.csv")

    return f