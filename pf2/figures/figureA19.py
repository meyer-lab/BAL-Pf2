"""
Figure A19
"""

import anndata
import pandas as pd
from pacmap import PaCMAP
from ..figures.common import getSetup, subplotLabel
import scanpy as sc
from ..figures.commonFuncs.plotPaCMAP import plot_labels_pacmap
from .figureA11 import add_obs_cmp_both_label, add_obs_label



def makeFigure():
    ax, f = getSetup((8, 8), (1, 1))
    subplotLabel(ax)
    cmp1 = 3; cmp2 = 26

    X = anndata.read_h5ad("prolifX.h5ad")
    X = add_obs_label(X, cmp1, cmp2)
    
    
    # cmp1 = 3; cmp2 = 26
    # pos1 = True; pos2 = True
    # threshold = 0.5
    # X = add_obs_cmp_both_label(X, cmp1, cmp2, pos1, pos2, top_perc=threshold)
    
    # X = X[((X.obsm["X_pf2_PaCMAP"][:, 0] < 5) & 
    #       (X.obsm["X_pf2_PaCMAP"][:, 0] > -15) &
    #        (X.obsm["X_pf2_PaCMAP"][:, 1] < -10)), :]

    # print(X)

    # pcm = PaCMAP(random_state=0)
    # X.obsm["X_pf2_PaCMAP"] = pcm.fit_transform(X.obsm["projections"]) 
    # sc.pp.neighbors(X, use_rep="projections", random_state=0)
    # sc.tl.leiden(X, random_state=0, resolution=.9)
    
    # print(X)
    # print(X.obs["leiden"].unique())
    # X.write("prolifX.h5ad")
    plot_labels_pacmap(X, "Label", ax[0])


    return f

