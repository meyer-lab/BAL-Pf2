"""Figure BS_A1: Description"""

import anndata
from .common import subplotLabel, getSetup
from ..data_import import condition_factors_meta
from ..tensor import correct_conditions
import numpy as np
import seaborn as sns
import pandas as pd
from pacmap import PaCMAP

def makeFigure():
    ax, f = getSetup((10, 10), (3, 3))
    subplotLabel(ax)

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    X.uns["Pf2_A"] = correct_conditions(X)
    cond_factors_df = condition_factors_meta(X)
    
    factors_only = [f"Cmp. {i}" for i in np.arange(1, X.uns["Pf2_A"].shape[1] + 1)]
    factors_only_df = cond_factors_df[factors_only]
    
    pcm = PaCMAP(n_components=10)
    factors_only_umap = pcm.fit_transform(factors_only_df) 
    # print(factors_only_umap)
    print(np.shape(factors_only_umap))
    
    umap_df = pd.DataFrame(data=factors_only_umap, columns=["PaCMAP1", "PaCMAP2"])
    
    sns.scatterplot(data=umap_df, x="PaCMAP1", y="PaCMAP2", ax=ax[0])
    
    umap_df["binary_outcome"] = cond_factors_df["binary_outcome"].to_numpy()
    

    print(umap_df)
    
    sns.scatterplot(data=umap_df, x="PaCMAP1", y="PaCMAP2", hue="binary_outcome", ax=ax[1])
    
    

    
  
    return f
