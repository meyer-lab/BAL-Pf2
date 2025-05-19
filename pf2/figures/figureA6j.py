"""Figure S1: Cell type abundance and distribution across patient statuses"""
import anndata
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotGeneral import rotate_xaxis
from ..data_import import add_obs, combine_cell_types
from ..utilities import bal_combine_bo_covid, cell_count_perc_df


"""Figure S1: Cell type abundance and distribution across patient statuses"""

import anndata
import numpy as np
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import pandas as pd
from matplotlib.axes import Axes
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotGeneral import rotate_xaxis
from ..data_import import add_obs, combine_cell_types
from ..utilities import bal_combine_bo_covid, cell_count_perc_df

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((7, 7), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    combine_cell_types(X)
    
    X = X[~X.obs["patient_category"].isin(["Non-Pneumonia Control", "COVID-19"])]

    celltype = ["combined_cell_type"]
    types = ["Cell Count", "Cell Type Percentage"]
    
    axs = 0
    for i, celltypes in enumerate(celltype):
        for j, type in enumerate(types):
            celltype_count_perc_df = cell_count_perc_df(X, celltype=celltypes, include_control=False)
            new_df = celltype_count_perc_df.loc[celltype_count_perc_df["Cell Type"] == "Other"].copy().reset_index(drop=True)
            new_df["Cell Type"] = new_df["Cell Type"].astype(str)
            final_df = new_df.reset_index(drop=True)
            print(final_df)
            
            # Plot the boxplot
            sns.boxplot(
                data=final_df,
                x="Cell Type",
                y=type,
                hue="Status",
                showfliers=False,
                ax=ax[axs],
            )
            rotate_xaxis(ax[axs])
            
            # Perform statistical tests and add annotations
            if type == "Cell Count":
                # Perform t-test
                print()
                group1 = final_df[final_df["Status"] == "D-nC19"][type].values
                group2 = final_df[final_df["Status"] == "L-nC19"][type].values
                print(group1)
                
                t_stat, p_val = stats.ttest_ind(group1, group2)
                print(p_val)
                add_stat_annotation(ax[axs], p_val, test_type="t-test")
                
            elif type == "Cell Type Percentage":
                # Perform WLS test
                pval_df = wls_stats_comparison(final_df, type, "Status", "D-nC19")
                print(pval_df)
                add_stat_annotation(ax[axs], pval_df["p Value"].iloc[0], test_type="WLS")
            
            axs += 1

    return f

def wls_stats_comparison(df, column_comparison_name, category_name, status_name):
    """Calculates whether cells are statistically significantly different using WLS"""
    pval_df = pd.DataFrame()
    df["Y"] = 1
    df.loc[df[category_name] == status_name, "Y"] = 0
    
    Y = df[column_comparison_name].to_numpy()
    X = df["Y"].to_numpy()
    weights = np.power(df["Cell Count"].values, 1)
    
    mod_wls = sm.WLS(Y, sm.tools.tools.add_constant(X), weights=weights)
    res_wls = mod_wls.fit()
    
    print(res_wls.pvalues)
    
    pval_df = pd.DataFrame({
        "Cell Type": ["Other"],
        "p Value": [res_wls.pvalues[1]]
    })
    
    return pval_df

def add_stat_annotation(ax, p_value, test_type):
    """Adds statistical annotation to the plot"""
    # Get current y-axis limits
    y_min, y_max = ax.get_ylim()
    
    # Determine significance level
    if p_value < 0.001:
        stars = '***'
    elif p_value < 0.01:
        stars = '**'
    elif p_value < 0.05:
        stars = '*'
    else:
        stars = 'ns'
    
    # Position the annotation above the plot
    text_y = y_max + 0.1 * (y_max - y_min)
    
    # Add the annotation
    ax.text(0.5, text_y, 
            f"{test_type}: p = {p_value:.3f} {stars}", 
            ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.8))
    
    # Adjust y-axis limits to accommodate the annotation
    ax.set_ylim(y_min, text_y + 0.1 * (y_max - y_min))


