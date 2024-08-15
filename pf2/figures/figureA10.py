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
    
    wprojs = X.obsm["weighted_projections"]
    thres_value = 5
    threshold = np.percentile(wprojs, thres_value, axis=0) 

    wprojs_subset = np.zeros((wprojs[wprojs[:, 0] < threshold[0], :].shape[0], wprojs.shape[1]))

    for i in range(len(threshold)):
        wprojs_subset[:, i] = wprojs[wprojs[:, i] < threshold[i], i]
        
        
    print

    return f

    
