"""Figure A2: Weighted projections for each component"""

from anndata import read_h5ad
from .common import getSetup
from .commonFuncs.plotPaCMAP import plot_wp_pacmap, plot_wp_per_celltype 
import numpy as np
import anndata



def makeFigure():
    ax, f = getSetup((5, 5), (2, 2))
    # subplotLabel(ax)

    X = read_h5ad("/opt/northwest_bal/full_fitted.h5ad", backed="r")
    
    cmp1 = 3
    cmp2 = 26
    threshold = .5
    X = add_cmp_both_label(X, cmp1, cmp2, pos1=True, pos2=True, top_perc=threshold)
    
    

    X_cmp1 = X[(X.obs[f"Cmp{cmp1}"] == True)]
    X_cmp2 = X[(X.obs[f"Cmp{cmp2}"] == True)]
    # for i in range(1, 51):
    plot_wp_pacmap(X_cmp1, cmp1, ax[0], cbarMax=0.3)
    plot_wp_pacmap(X_cmp2, cmp2, ax[1], cbarMax=0.3)
        # plot_wp_per_celltype(X, i, ax[i-1])

    return f




def add_cmp_both_label(X: anndata.AnnData, cmp1: int, cmp2: int, pos1=True, pos2=True, top_perc=1):  
    """Adds if cells in top/bot percentage""" 
    wprojs = X.obsm["weighted_projections"]
    pos_neg = [pos1, pos2]
    
    for i, cmp in enumerate([cmp1, cmp2]):
        if i == 0:
            if pos_neg[i] is True:    
                thres_value = 100-top_perc
                threshold1 = np.percentile(wprojs, thres_value, axis=0) 
                idx = wprojs[:, cmp-1] > threshold1[cmp-1]
                
            else:
                thres_value = top_perc
                threshold1 = np.percentile(wprojs, thres_value, axis=0) 
                idx = wprojs[:, cmp-1] < threshold1[cmp-1]
                
        if i == 1:
            if pos_neg[i] is True:    
                thres_value = 100-top_perc
                threshold2 = np.percentile(wprojs, thres_value, axis=0) 
                idx = wprojs[:, cmp-1] > threshold2[cmp-1]
            else:
                thres_value = top_perc
                threshold2 = np.percentile(wprojs, thres_value, axis=0) 
                idx = wprojs[:, cmp-1] < threshold1[cmp-1]
                
        
        X.obs[f"Cmp{cmp}"] = idx

    if pos1 and pos2 is True:
        idx = (wprojs[:, cmp1-1] > threshold1[cmp1-1]) & (wprojs[:, cmp2-1] > threshold2[cmp2-1])
    elif pos1 and pos2 is False:
        idx = (wprojs[:, cmp1-1] < threshold1[cmp1-1]) & (wprojs[:, cmp2-1] < threshold2[cmp2-1])
    elif pos1 is True & pos2 is False:
        idx = (wprojs[:, cmp1-1] > threshold1[cmp1-1]) & (wprojs[:, cmp2-1] < threshold2[cmp2-1])
    elif pos1 is False & pos2 is True:
        idx = (wprojs[:, cmp1-1] < threshold1[cmp1-1]) & (wprojs[:, cmp2-1] > threshold2[cmp2-1])
        
    X.obs["Both"] = idx
    
    return X
