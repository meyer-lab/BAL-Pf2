"""Figure S8: Metadata correlation analysis"""

import anndata
from .common import (
    subplotLabel,
    getSetup,
)
from .commonFuncs.plotGeneral import plot_correlation_heatmap
from ..data_import import meta_raw_df
from ..correlation import correlation_df



def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((15, 15), (2, 2))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")

    all_meta_df = meta_raw_df(X, all=True)
    c19_meta_df, nc19_meta_df = meta_raw_df(X, all=False)
    
    meta = [all_meta_df, c19_meta_df, nc19_meta_df]
    
    for i, meta_df in enumerate(meta):
        corr_df = correlation_df(meta_df, meta=True)
        plot_correlation_heatmap(corr_df, xticks=corr_df.columns, 
                                yticks=corr_df.columns, ax=ax[i], mask=True)
        
    labels = ["All", "C19", "nC19"]
    for i in range(3):
        ax[i].set(title=labels[i])

    return f

