"""
Figure A18:
"""

import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.axes import Axes
import anndata
from .common import subplotLabel, getSetup
from ..figures.commonFuncs.plotGeneral import bal_combine_bo_covid, rotate_xaxis, add_obs_cmp_both_label, add_obs_label
from ..data_import import add_obs, combine_cell_types
from .commonFuncs.plotFactors import bot_top_genes
import scanpy as sc
from ..data_import import add_obs, condition_factors_meta
from .figureA11 import add_obs_cmp_both_labelm 

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # ax, f = getSetup((8, 12), (7, 4))
    ax, f = getSetup((18, 18), (7, 4))
    
    subplotLabel(ax)
    
    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    add_obs(X, "binary_outcome")
    add_obs(X, "patient_category")
    combine_cell_types(X)
    
    cmp1 = 3
    cmp2 = 26
    threshold = 1
    X = add_obs_cmp_both_label(X, cmp1, cmp2, pos1=False, pos2=False, top_perc=threshold)
    X = add_obs_label(X, cmp1, cmp2)
    
    X.obs["Label"] = X.obs["Cell Type"].astype(str) + X.obs["Label"].astype(str)

    XX = X[(X.obs["Label"] == "MacrophagesCmp3")]
    XXXX = X[(X.obs["Label"] == "MacrophagesCmp26")]
    XXX = X[(X.obs["Label"] == "MacrophagesNoLabel")]
    smallX = sc.pp.subsample(XXX, fraction=.1, copy=True) 

    finalX = anndata.concat([smallX, XX, XXXX])

    # sc.tl.rank_genes_groups(X, "Label", method="wilcoxon", groups="MacrophagesCmp3", reference="MacrophagesCmp26")
    # sc.pl.rank_genes_groups(X, n_genes=30, save="Cmp_3_26_wilcoxon.png")

    sc.tl.rank_genes_groups(finalX, "Label", method="wilcoxon", groups=["MacrophagesCmp3", "MacrophagesCmp26"], reference="MacrophagesNoLabel")
    sc.pl.rank_genes_groups(finalX, n_genes=30, save="Cmp_3_26_test.png")

    return f

