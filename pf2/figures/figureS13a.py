"""Figure 13a"""

import anndata
from ..gene_analysis import gsea_analysis_per_cmp


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    gsea_analysis_per_cmp(X, 22, output_file="output/figureS13a.svg")
    
    return
    