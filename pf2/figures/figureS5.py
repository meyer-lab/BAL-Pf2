"""
Figure S5:
"""
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

import anndata
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import spearmanr
from matplotlib.axes import Axes
from .common import getSetup
from ..data_import import add_obs, combine_cell_types, condition_factors_meta, condition_factors_meta_raw
from .commonFuncs.plotGeneral import rotate_xaxis
from ..utilities import cell_count_perc_df


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # ax, f = getSetup((20, 20), (2, 2))
    # ax, f = getSetup((8, 8), (2, 2))
    ax, f = getSetup((7, 7), (2, 2))

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    factors_meta_df, _ = condition_factors_meta_raw(X)
    factors_meta_df = factors_meta_df.reset_index()
    X = add_obs(X, "binary_outcome")
    X = add_obs(X, "patient_category")
    X = X[~X.obs["patient_category"].isin(["Non-Pneumonia Control", "COVID-19"])]
    
    combine_cell_types(X)
    celltype_count_perc_df = cell_count_perc_df(X, celltype="cell_type")
    
    #### Plot scatter plot of B cells vs pDC percentages
 
    df = celltype_count_perc_df.loc[celltype_count_perc_df["Cell Type"].isin(["B cells", "pDC"])]
    axs = 0
    for i, column in enumerate(["Cell Count", "Cell Type Percentage"]):
        merged_df = pd.merge(
            df,
            factors_meta_df[["sample_id", "Cmp. 22", "Cmp. 62", "icu_day", "immunocompromised_flag", "episode_etiology"]],
            on="sample_id",
            how="inner"
        )
        # Convert Immunocompromised flag to be yes/no 
        merged_df["AIC"] = merged_df["immunocompromised_flag"].replace({1: "Yes", 0: "No"})
        # Merge icu days into categroy 
        # merged_df["icu_day"] = pd.cut(
        #     merged_df["icu_day"],
        #     bins=[1, 7, 27, 100],
        #     labels=["1-7", "8-27", "27+"]
        # )
        # sns.scatterplot(merged_df, x="Cmp. 22", y="Cmp. 62", hue="Status", style="icu_day", ax=ax[axs])
        # ax[axs].set_title(f"pearson: {pearsonr(merged_df["Cmp. 22"], merged_df["Cmp. 62"])[0]:.2f}")
        
        # print(merged_df)
        # a
        plot_celltype_scatter(merged_df=merged_df, columns=column, celltype1="B cells", celltype2="pDC", otherlabel="AIC", ax=ax[axs])
        axs += 1

    
    #### Plot stripplot of cell counts pDC/ B cells
    # axs=0
    # for i, celltype in enumerate(["B cells", "pDC"]):
    #     for j, type in enumerate(["Cell Count", "Cell Type Percentage"]):
    #         df = celltype_count_perc_df.loc[celltype_count_perc_df["Cell Type"] == celltype]
    #         merged_df = pd.merge(
    #             df,
    #             factors_meta_df[["sample_id"]],
    #             on="sample_id",
    #             how="inner"
    #         )
    #         sns.stripplot(
    #             data=merged_df,
    #             x="Status",
    #             y=type,
    #             hue="Status",
    #             dodge=True,
    #             ax=ax[axs],
    #         )
    #         ax[axs].set_title(f"{celltype} {type}")
    #         axs += 1
    
    ### Plot correlation of component weights and cell type percentage for pDC/B cells
    # axs=0
    # for i, celltype in enumerate(["B cells", "pDC"]):
    #     for j, type in enumerate(["Cell Type Percentage"]):
    #         df = celltype_count_perc_df.loc[celltype_count_perc_df["Cell Type"] == celltype]
    #         merged_df = pd.merge(
    #             df,
    #             factors_meta_df[["sample_id"] + [f"Cmp. {i+1}" for i in range(80)]],
    #             on="sample_id",
    #             how="inner"
    #         )
    #         plot_correlation_all_cmps(merged_df=merged_df, ax=ax[axs], cellPerc=(type == "Cell Type Percentage"), celltype=celltype)
    #         axs += 1



    return f

def plot_celltype_scatter(
    merged_df: pd.DataFrame,
    columns: str, 
    celltype1: str,
    celltype2: str,
    otherlabel: str,
    ax: Axes,
):
    """Plots a scatter plot of cell percentages for two cell types, labeled by Status."""
    # Filter and merge data
    df1 = merged_df[merged_df["Cell Type"] == celltype1].rename(
        columns={columns: f"{celltype1} {columns}"}
    )
    df2 = merged_df[merged_df["Cell Type"] == celltype2].rename(
        columns={columns: f"{celltype2} {columns}"}
    )

    scatter_df = pd.merge(
        df1[["sample_id", f"{celltype1} {columns}", "Status", otherlabel]],
        df2[["sample_id", f"{celltype2} {columns}", "Status"]],
        on=["sample_id", "Status"]
    )
    
    # Create scatter plot (no log scale)
    sns.scatterplot(
        data=scatter_df,
        x=f"{celltype1} {columns}",
        y=f"{celltype2} {columns}",
        hue="Status",
        style=otherlabel,
        ax=ax,
    )

    # # Prepare data for SVM (no log transformation)
    # X = scatter_df[[f"{celltype1} {columns}", f"{celltype2} {columns}"]].values
    # y = (scatter_df[otherlabel] == "Yes").astype(int)
    
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    
    # # Fit SVM with automatic gamma scaling and balanced class weights
    # clf = SVC(kernel='linear', C=1.0, class_weight='balanced')
    # clf.fit(X_scaled, y)
    
    # # Create grid for decision boundary visualization
    # x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
    #                     np.linspace(y_min, y_max, 100))
    
    # # Transform grid points and get predictions
    # Z = clf.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
    # Z = Z.reshape(xx.shape)
    
    # # Plot decision boundary and margins
    # ax.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.5, 
    #             linestyles=['--'], linewidths=2)
    

    # ax.legend()
    # Add labels and title
    ax.set_xlabel(f"{celltype1} Percentage")
    ax.set_ylabel(f"{celltype2} Percentage")
    spearman = spearmanr(X[:, 0], X[:, 1])
    ax.set_title(f"Spearman: {spearman[0]:.2f} Pvalue: {spearman[1]:.2e}")
    
    ax.set_xscale("log")
    ax.set_yscale("log")

    return scatter_df
  