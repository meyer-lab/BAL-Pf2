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
    ax, f = getSetup((4, 4), (2, 2))

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
    for i, column in enumerate(["Cell Type Percentage"]):
        merged_df = pd.merge(
            df,
            factors_meta_df[["sample_id", "Cmp. 22", "Cmp. 62", "icu_day", "immunocompromised_flag", "episode_etiology"]],
            on="sample_id",
            how="inner"
        )
        # Convert Immunocompromised flag to be yes/no 
        merged_df["AIC"] = merged_df["immunocompromised_flag"].replace({1: "Yes", 0: "No"})
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
    
    # ### Plot correlation of component weights and cell type percentage for pDC/B cells
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
    
    
    pal = sns.color_palette()
    pal = [pal[1], pal[3]]
    pal = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' for r, g, b in pal]
    # Create scatter plot (no log scale)
    sns.scatterplot(
        data=scatter_df,
        x=f"{celltype1} {columns}",
        y=f"{celltype2} {columns}",
        hue="Status",
        style=otherlabel,
        palette=pal,
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
    spearman = spearmanr(scatter_df[f"{celltype1} {columns}"], scatter_df[f"{celltype2} {columns}"])
    ax.set_title(f"Spearman: {spearman[0]:.2f} Pvalue: {spearman[1]:.2e}")
    
    ax.set(xlim=(0.005, 10), ylim=(0.005, 10))
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    # Set axis limits

    return scatter_df
  
  
  
  
def plot_correlation_all_cmps(
    merged_df: pd.DataFrame,
    ax: Axes,
    cellPerc=True,
    celltype: str = "pDC",
):
    """Plot Pearson correlation of all component weights and cell type percentage, split by Status."""
    # Determine the column to use for cell percentage or count
    cellPercCol = "Cell Type Percentage" if cellPerc else "Cell Count"

    # Prepare the dataframe for correlation analysis
    correlationdf = pd.DataFrame([])
    for cmp in range(1, 81):  # Iterate over all components (Cmp. 1 to Cmp. 80)
        cmp_col = f"Cmp. {cmp}"

        for status in merged_df["Status"].unique():
            # Filter data for the specific status
            status_df = merged_df[merged_df["Status"] == status]
            status_df = status_df[status_df["Cell Type"] == celltype]

            # Calculate Pearson correlation
            if len(status_df) > 1:  # Ensure there are enough data points
                spearman_corr, pval = spearmanr(status_df[cmp_col], status_df[cellPercCol])
            else:
                spearman_corr = np.nan  # Not enough data points

            # Append the result to the correlation dataframe
            correlationdf = pd.concat([
                correlationdf,
                pd.DataFrame({
                    "Component": [cmp],
                    "Status": [status],
                    "Correlation": [spearman_corr],
                    "P-value": [pval]
                })
            ])

    # Only keep the 10 components with the highest absolute correlation
    correlationdf = correlationdf.loc[
        correlationdf["Component"].isin(
            correlationdf.groupby("Component")["Correlation"]
            .apply(lambda x: x.abs())
            .nlargest(10)
            .index.get_level_values(0)
        )
    ]

    print(correlationdf)
    sns.barplot(
        data=correlationdf,
        x="Component",
        y="Correlation",
        hue="Status",
        ax=ax
    )
    ax.set_ylim(-.75, 1)
    rotate_xaxis(ax)
    ax.set(
        title=f"{cellPercCol}",
        ylabel="Spearman Correlation",
        xlabel="Component"
    )