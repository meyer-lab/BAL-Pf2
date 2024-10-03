"""
Figure A19
"""

import anndata
import pandas as pd
from pacmap import PaCMAP
from ..figures.common import getSetup, subplotLabel
import scanpy as sc
from ..figures.commonFuncs.plotPaCMAP import plot_labels_pacmap



def makeFigure():
    ax, f = getSetup((8, 8), (1, 1))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    
    X = X[((X.obsm["X_pf2_PaCMAP"][:, 0] < 5) & 
          (X.obsm["X_pf2_PaCMAP"][:, 0] > -15) &
           (X.obsm["X_pf2_PaCMAP"][:, 1] < -10)), :]

    print(X)

    pcm = PaCMAP(random_state=0)
    X.obsm["X_pf2_PaCMAP"] = pcm.fit_transform(X.obsm["projections"]) 
    sc.pp.neighbors(X, use_rep="projections", random_state=0)
    sc.tl.leiden(X, random_state=0, resolution=.5)
    
    print(X)
    X.write("prolifX.h5ad")
    plot_labels_pacmap(X, "leiden", ax[0])


    return f

