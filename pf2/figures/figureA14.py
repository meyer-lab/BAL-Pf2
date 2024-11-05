"""
Figure A14:
"""
from .commonFuncs.plotPaCMAP import plot_labels_pacmap
from ..data_import import combine_cell_types, add_obs
import anndata
from .common import subplotLabel, getSetup
from ..figures.commonFuncs.plotGeneral import add_obs_cmp_both_label, add_obs_label
import seaborn as sns
import matplotlib.colors as mcolors
import numpy as np

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((6, 6), (2, 2))

    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    combine_cell_types(X)

    cmp1 = 2; cmp2 = 25
    pos1 = True; pos2 = False
    threshold = 0.5
    X = add_obs_cmp_both_label(X, cmp1, cmp2, pos1, pos2, top_perc=threshold)
    print(np.count_nonzero(X.obs["Cmp2"]))
    print(np.count_nonzero(X.obs["Cmp25"]))
    print(np.count_nonzero(X.obs["Both"]))
    
    X = add_obs_label(X, cmp1, cmp2)
    print(np.unique(X.obs["Label"], return_counts=True))
    
    colors = ["black", "fuchsia", "turquoise", "gainsboro"]
    pal = []
    for i in colors:
        pal.append(mcolors.CSS4_COLORS[i])
        
    print(np.unique(X.obs["Label"], return_counts=True))
    plot_labels_pacmap(X, "Label", ax[0], color_key=pal)
    
    
    return f


