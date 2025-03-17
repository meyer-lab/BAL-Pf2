"""Figure 13b"""

import anndata
from ..gene_analysis import gsea_overrep_per_cmp


def makeFigure():
    """Get a list of the axis objects and create a figure."""

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    gsea_overrep_per_cmp(X, 3, pos=True, enrichr=False, output_file="output/figureS13b.svg")
    
    return
