# """
# Lupus: Cell type percentage between status (with stats comparison) and
# correlation between component and cell count/percentage for each cell type
# """
# import anndata
# from pf2.figures.common import getSetup, subplotLabel
# from pf2.tensor import correct_conditions

# from pf2.data_import import  add_obs
# from anndata import read_h5ad
# from .common import (
#     subplotLabel,
#     getSetup,
# )
# import numpy as np
# import seaborn as sns
# import pandas as pd
# from scipy.stats import pearsonr
# from .commonFuncs.plotGeneral import rotate_xaxis
# from matplotlib.axes import Axes
# import anndata


# def makeFigure():
#     """Get a list of the axis objects and create a figure."""
#     # Get list of axis objects
#     ax, f = getSetup((30, 30), (5, 10))
    

#     # Add subplot labels
#     subplotLabel(ax)
    
#     X = read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
#     X = add_obs(X, "binary_outcome")
#     X = add_obs(X, "patient_category")
#     X.uns["Pf2_A"] = correct_conditions(X)


#     return f

# def makeFigure():
#     full_makeup /= full_makeup.sum()
#     full_makeup = full_makeup.drop("Ionocytes")

#     axs, fig = getSetup((20, 40), (10, 5))

#     for ax, comp in zip(axs, components):
#         comp_makeup = (
#             data.obs.iloc[
#                     -50000:
#                 ],
#                 :,
#             ]
#             .loc[:, "cell_type"]
#             .value_counts()
#         )
#         diff = abs(comp_makeup - full_makeup)

#     components = np.arange(data.obsm["projections"].shape[1]
# ) + 1

#     for ax, comp in zip(axs, components):
#         comp_makeup = (
#             data.obs.iloc[
#                 np.argpartition(data.obsm["projections"][:,
# comp - 1], -50000)[
#                     -50000:
#                 ],
#                 :,
#             ]
#             .loc[:, "cell_type"]
#             .value_counts()
#         )
#         comp_makeup /= comp_makeup.sum()
#         comp_makeup = comp_makeup.loc[full_makeup.index]
#         diff = abs(comp_makeup - full_makeup)
#         scaled_diff = diff / full_makeup

#         comp_makeup = comp_makeup.loc[
#             np.logical_or(diff > 0.05, scaled_diff > 1)
#         ]
#         _full_makeup = full_makeup.loc[
#             np.logical_or(diff > 0.05, scaled_diff > 1)
#         ]

#         ax.bar(
#             np.arange(0, len(_full_makeup) * 3, 3),
#             _full_makeup,
#             label="All Cells",
#             color="tab:blue",
#         )
#         ax.bar(
#             np.arange(1, len(comp_makeup) * 3, 3),
#             comp_makeup,
#             label=f"Component {comp} Cells",
#             color="tab:orange",
#         )

#         ax.legend()
#         ax.set_xticks(np.arange(0.5, len(comp_makeup) * 3, 3
# ))
#         ax.set_xticklabels(comp_makeup.index, rotation=90)
#         ax.set_ylabel("Cell Proportion")

#         ax.set_title(f"Component {comp}")

#     return fig