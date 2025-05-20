"""Figure A1d_g"""

import anndata
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotGeneral import rotate_xaxis
from ..data_import import add_obs, condition_factors_meta, combine_cell_types


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((8, 8), (3, 3))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    print(X)
    # add_obs(X, "binary_outcome")
    # add_obs(X, "patient_category")
    # combine_cell_types(X)

    # cond_fact_meta_df = condition_factors_meta(X)
    
    # plot_sample_count(cond_fact_meta_df, ax[0], ax[1], combine_categories=True, include_control=False)
    # plot_sample_count(cond_fact_meta_df, ax[2], ax[3], combine_categories=False, include_control=True)

    # cond_fact_meta_df = cond_fact_meta_df.drop_duplicates(subset=["patient_id"])
    # plot_sample_count(cond_fact_meta_df, ax[4], ax[5], combine_categories=True, include_control=False)
    # plot_sample_count(cond_fact_meta_df, ax[6], ax[7], combine_categories=False, include_control=True)
        
    # for i in [1, 3]:
    #     ax[i].set(ylabel="Sample Proportion")
    # for i in [4, 6]:
    #     ax[i].set(ylabel="Patient Count")
    # for i in [5, 7]:
    #     ax[i].set(ylabel="Patient Proportion")

    return f



def plot_sample_count(
    df: pd.DataFrame,
    ax1: Axes,
    ax2: Axes,
    combine_categories=True,
    include_control=True,
):
    """Plots overall patients in each category."""
    if include_control is False:
            df = df[df["patient_category"] != "Non-Pneumonia Control"]
      
    if combine_categories is True:
        comparison_column = "Status"
    else:
        comparison_column = "Uncombined"
        
    dfCond = (
        df.groupby(comparison_column, observed=True).size().reset_index(name="Sample Count")
    )

    if combine_categories is True: 
        sns.barplot(data=dfCond, x=comparison_column, y="Sample Count", hue="Status", ax=ax1)
    else: 
        sns.barplot(data=dfCond, x=comparison_column, y="Sample Count", color="k", ax=ax1)
        rotate_xaxis(ax1)

    total = dfCond["Sample Count"].sum()
    dfCond["Sample Count"] = dfCond["Sample Count"] / total

    if combine_categories is True: 
        sns.barplot(data=dfCond, x=comparison_column, y="Sample Count", hue="Status", ax=ax2)
    else:
        sns.barplot(data=dfCond, x=comparison_column, y="Sample Count", color="k", ax=ax2)
        rotate_xaxis(ax2)
    

