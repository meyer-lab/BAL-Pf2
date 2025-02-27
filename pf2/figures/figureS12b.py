"""Figure S12a"""

import anndata
from .common import (
    getSetup,
)
from ..data_import import meta_raw_df, bal_combine_bo_covid
import seaborn as sns
from .commonFuncs.plotGeneral import rotate_xaxis
from ..correlation import meta_groupings

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((15, 15), (5, 5))

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")

    all_meta_df = meta_raw_df(X, all=True)
    all_meta_df = bal_combine_bo_covid(all_meta_df)
    # all_meta_df = all_meta_df[all_meta_df["patient_category"] != "Non-Pneumonia Control"]

    for i, corr in enumerate(meta_groupings):
        counts = all_meta_df.groupby([corr, "Status"]).size().reset_index(name="Count")
        total_counts = counts.groupby(corr)["Count"].transform("sum")
        counts["Percentage"] = counts["Count"] / total_counts * 100
        sns.barplot(counts, x=corr, y="Percentage",hue="Status", ax=ax[i])
        rotate_xaxis(ax[i], rotation=90)

    return f
