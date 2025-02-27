"""Figure S12a"""

import anndata
from .common import (
    getSetup,
)
from ..data_import import meta_raw_df, bal_combine_bo_covid
import seaborn as sns
from .commonFuncs.plotGeneral import rotate_xaxis
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
        # Calculate counts
        counts = all_meta_df.groupby([corr, "Status"]).size().reset_index(name='Count')
        
        # Calculate percentages
        total_counts = counts.groupby(corr)['Count'].transform('sum')
        counts['Percentage'] = counts['Count'] / total_counts * 100
        
        # Format the values to show percentages with counts in parentheses
        counts['Formatted'] = counts.apply(lambda row: f"{row['Percentage']:.1f}% ({row['Count']})", axis=1)
        
        # Pivot the table to have categories as rows and statuses as columns
        pivot_table = counts.pivot(index=corr, columns='Status', values='Formatted')
        
        # Add a column to identify the category
        pivot_table['Category'] = corr + f" ({np.sum(total_counts.unique())})"
        
        pivot_table.index = [f"{idx} ({total_counts[counts[corr] == idx].iloc[0]})" for idx in pivot_table.index]
        
        # Append to the combined DataFrame
        combined_df = pd.concat([combined_df, pivot_table], axis=0)

    combined_df = combined_df.reset_index().rename(columns={'index': 'Category_Index'})
    combined_df = combined_df[['Category', 'Category_Index'] + [col for col in combined_df.columns if col not in ['Category', 'Category_Index']]]
    
    print(combined_df)
    
    print(pivot_table)
    # for i, corr in enumerate(meta_groupings):
    #     counts = all_meta_df.groupby([corr, "Status"]).size().reset_index(name="Count")
    #     total_counts = counts.groupby(corr)["Count"].transform("sum")
    #     counts["Percentage"] = counts["Count"] / total_counts * 100
    #     sns.barplot(counts, x=corr, y="Percentage",hue="Status", ax=ax[i])
    #     rotate_xaxis(ax[i], rotation=90)
    
    

    return f


meta_groupings = [
    "ecmo_flag", "episode_category", "episode_etiology",
    "pathogen_virus_detected", "pathogen_bacteria_detected",
    "pathogen_fungi_detected", "smoking_status", "icu_stay",
    "admission_source_name", "global_cause_failure", "patient_category",
    "covid_status", "gender", "tracheostomy_flag", "immunocompromised_flag",
    "norepinephrine_flag", "remdesivir_received", "episode_is_cured"
]
