"""Table S1"""

import anndata
from .common import getSetup
from ..data_import import meta_raw_df, bal_combine_bo_covid
from ..correlation import meta_correlates
import pandas as pd


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((15, 15), (7, 7))

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")

    all_meta_df = meta_raw_df(X, all=True)
    all_meta_df = bal_combine_bo_covid(all_meta_df)
    # all_meta_df = all_meta_df[all_meta_df["patient_category"] != "Non-Pneumonia Control"]

    total_df = pd.DataFrame([])
    for i, corr in enumerate(meta_correlates):
        corr_df = all_meta_df[["Status", corr]].dropna()
        for j, status in enumerate(corr_df["Status"].unique()):
            status_df = corr_df[corr_df["Status"] == status]
            temp_df = pd.DataFrame({
                "Mean": [status_df[corr].mean()],
                "Std": [status_df[corr].std()],
                "MetaData": [corr],
                "Status": [status],
                "Count": [status_df.shape[0]]
            })
            total_df = pd.concat([total_df, temp_df], ignore_index=True)

    complete_df = create_comprehensive_table(total_df, total_df["MetaData"].unique())
    print(complete_df)
    complete_df.to_csv("pf2/data/meta_correlates_summary.csv")

    return f


def create_comprehensive_table(df, metrics_list):
    """Combines percetanges, mean and std into a comprehensive table."""
    results = {}
    for metric in metrics_list:
        metric_data = df[df['MetaData'] == metric]
        if len(metric_data) > 0:
            grouped = metric_data.groupby('Status')
            total_count = metric_data['Count'].sum()
            for name, group in grouped:
                if name not in results:
                    results[name] = {}
                count = group['Count'].iloc[0]
                percentage = (count / total_count) * 100
                mean_std = f"{group['Mean'].iloc[0]:.1f} ± {group['Std'].iloc[0]:.1f}"
                results[name][f"{metric} ({total_count})"] = f"{mean_std} ({count}, {percentage:.1f}%)"
    
    result_df = pd.DataFrame.from_dict(results, orient='index')
    
    return result_df.transpose()