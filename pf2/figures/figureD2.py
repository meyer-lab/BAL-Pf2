"""Figure Debug 1: Factor Matrices & Cell Embedding"""
import time

import datashader as ds
import datashader.transfer_functions as tf
from matplotlib import colormaps
from matplotlib.colors import to_hex
import pandas as pd
import seaborn as sns
import numpy as np
from pf2.data_import import import_data
from pf2.figures.common import ds_show, get_canvas, getSetup, plot_gene_factors
from pf2.tensor import pf2, correct_conditions
import scanpy as sc
from matplotlib.patches import Patch
from matplotlib import pyplot as plt
from pf2.figures.commonFuncs.plotFactors import plot_condition_factors, plot_gene_factors, plot_eigenstate_factors
from pf2.figures.commonFuncs.plotPaCMAP import plot_labels_pacmap


def makeFigure():
    ax, f = getSetup((12, 4), (1, 4))
    
    start = time.time()
    data = import_data()
    data = sc.pp.subsample(data, fraction=.05, random_state=1, copy=True) 
    print(data)
    X, _ = pf2(data, rank=5)
    print(f"Factorization Time: {time.time() - start}")
    
    

    print(X)
    
    
    # data.uns["Pf2_A"] = correct_conditions(data)
    plot_condition_factors(X, ax[0])
    plot_eigenstate_factors(X, ax[1])
    plot_gene_factors(X, ax[2])
    plot_labels_pacmap(X, "cell_type", ax[3])



    # embedding = pd.DataFrame(
    #     factors.obsm["embedding"], index=data.obs.index, columns=["x", "y"]
    # )
    # embedding.loc[:, "label"] = X.obs.loc[:, "cell_type"].values
    # embedding.loc[:, "label"] = (
    #     embedding.loc[:, "label"].replace(CONVERSIONS).values
    # )
   
    
    return f




CONVERSIONS = {
    "CD8 T cells": "T Cells",
    "Monocytes1": "Monocytes",
    "Mac3 CXCL10": "Macrophages",
    "Monocytes2": "Monocytes",
    "B cells": "B Cells",
    "CD4 T cells": "T Cells",
    "CM CD8 T cells": "T Cells",
    "Tregs": "T-regulatory",
    "Plasma cells1": "B Cells",
    "Migratory DC CCR7": "Dendritic Cells",
    "Proliferating T cells": "Proliferating",
    "Monocytes3 HSPA6": "Monocytes",
    "Mac2 FABP4": "Macrophages",
    "DC2": "Dendritic Cells",
    "Mac4 SPP1": "Macrophages",
    "pDC": "Dendritic Cells",
    "Mac1 FABP4": "Macrophages",
    "Proliferating Macrophages": "Macrophages",
    "Mac6 FABP4": "Macrophages",
    "DC1 CLEC9A": "Dendritic Cells",
    "IFN resp. CD8 T cells": "T Cells",
    "NK/gdT cells": "NK Cells",
    "Mast cells": "Other",
    "Secretory cells": "Other",
    "Ciliated cells": "Other",
    "Epithelial cells": "Other",
    "Mac5 FABP4": "Macrophages",
    "Ionocytes": "Other",
}
SCATTER_COLORS = [to_hex(color) for color in colormaps["tab20"].colors]
COLORS = {
    "B Cells": "#ffbb78",
    "Dendritic Cells": "#98df8a",
    "Macrophages": "#ff7f0e",
    "Monocytes": "#aec7e8",
    "NK Cells": "#ff9896",
    "Other": "#9467bd",
    "Proliferating": "#d62728",
    "T Cells": "#1f77b4",
    "T-regulatory": "#2ca02c",
}