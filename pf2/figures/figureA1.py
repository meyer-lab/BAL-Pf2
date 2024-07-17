"""Figure A1: Condition, eigen-state, and gene factors, along with PaCMAP labeled by cell type"""

import anndata
from pf2.figures.common import getSetup, subplotLabel
from pf2.tensor import correct_conditions
from pf2.figures.commonFuncs.plotFactors import (
    plot_condition_factors,
    plot_gene_factors,
    plot_eigenstate_factors,
)
from pf2.figures.commonFuncs.plotPaCMAP import plot_labels_pacmap
from pf2.data_import import combine_cell_types, add_obs


def makeFigure():
    ax, f = getSetup((50, 50), (2, 3))
    subplotLabel(ax)
    
    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")

    X.uns["Pf2_A"] = correct_conditions(X)
    plot_condition_factors(X, ax[0], cond="sample_id")
    plot_eigenstate_factors(X, ax[1])
    plot_gene_factors(X, ax[2])
    plot_labels_pacmap(X, "cell_type", ax[3])
    
    add_obs(X, "patient_category")
    plot_labels_pacmap(X, "patient_category", ax[4])
    combine_cell_types(X)
    plot_labels_pacmap(X, "combined_cell_type", ax[5])

    return f
