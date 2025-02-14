"""Figure S12a"""

import anndata
from .common import (
    subplotLabel,
    getSetup,
)
from ..data_import import meta_raw_df, bal_combine_bo_covid
from ..correlation import correlates
import seaborn as sns
from .commonFuncs.plotGeneral import rotate_xaxis


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((15, 15), (7, 7))

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")

    all_meta_df = meta_raw_df(X, all=True)
    all_meta_df = bal_combine_bo_covid(all_meta_df)
    # all_meta_df = all_meta_df[all_meta_df["patient_category"] != "Non-Pneumonia Control"]

    for i, corr in enumerate(correlates):
        sns.violinplot(all_meta_df.sort_values("Status"), x="Status", y=corr, hue="Status", ax=ax[i])
        rotate_xaxis(ax[i], rotation=90)

    return f

