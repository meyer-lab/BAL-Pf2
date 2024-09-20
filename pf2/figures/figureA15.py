"""
Figure A14:
"""
from .commonFuncs.plotPaCMAP import plot_labels_pacmap
from ..data_import import combine_cell_types, add_obs
import anndata
from .common import subplotLabel, getSetup
from ..figures.commonFuncs.plotGeneral import add_obs_cmp_both_label_three, add_obs_label_three
import seaborn as sns
import matplotlib.colors as mcolors
import numpy as np

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((8, 8), (2, 2))

    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    combine_cell_types(X)


    cmp1 = 15; cmp2 = 16; cmp3 = 19
    pos1 = False; pos2 = True; pos3 = False
    threshold = 0.5
    
    X = add_obs_cmp_both_label_three(X, cmp1, cmp2, cmp3, pos1, pos2, pos3, top_perc=threshold)
    print(X)
    X = add_obs_label_three(X, cmp1, cmp2, cmp3)

    print(X)
    print(X.obs["Label"].to_numpy())
    print(np.unique(X.obs["Label"]))
    colors = ["black", "turquoise", "fuchsia", "slateblue", "gainsboro"]
    pal = []
    for i in colors:
        pal.append(mcolors.CSS4_COLORS[i])
        
        
    plot_labels_pacmap(X, "Label", ax[0], color_key=pal)
    
    
    # colors = ["black", "turquoise", "fuchsia", "gainsboro"]
    # pal = []
    # for i in colors:
    #     pal.append(mcolors.CSS4_COLORS[i])
        
    
    # cmp1 = 27; cmp2 = 46
    # pos1 = True; pos2 = True
    # X = add_obs_cmp_both_label(X, cmp1, cmp2, pos1, pos2, top_perc=threshold)
    # X = add_obs_label(X, cmp1, cmp2)
    
    # plot_labels_pacmap(X, "Label", ax[1], color_key=pal)
    
    
    return f


