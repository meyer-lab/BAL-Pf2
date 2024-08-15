"""
Figure A4: XXX
"""
import anndata
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from ..tensor import correct_conditions
from .common import getSetup

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((12, 12), (1, 1))

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
   

    condition_factors_df = pd.DataFrame(
        data=X.uns["Pf2_A"],
        columns=[f"Cmp. {i}" for i in np.arange(1, X.uns["Pf2_A"].shape[1] + 1)],
    )
    
    wProjs = X.obsm["weighted_projections"]
    

    thres_value = 5
    threshold = np.percentile(wProjs, thres_value, axis=0) 
    news = wProjs < threshold
    print(np.shape(news))
    
    a

    new_projs = np.zeros((wProjs[:]))
    for i in range(len(threshold)):
        new_projs = wProjs[wProjs[:, i] < threshold[i], :]
        print(np.shape(new_projs))
        
        
    # newx = s[:, s[0,:] < threshold]

#     print(newx)
#     print(np.shape(newx))


#    X = X[ind[:, cmp-1], :]




    return f

    
