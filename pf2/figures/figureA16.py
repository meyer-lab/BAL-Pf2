"""Figure A1: Condition, eigen-state, and gene factors,
along with PaCMAP labeled by cell type"""

import anndata
import pandas as pd
from ..tensor import correct_conditions
from ..data_import import combine_cell_types, add_obs
from ..figures.common import getSetup, subplotLabel
from ..figures.commonFuncs.plotFactors import (
    plot_condition_factors,
    plot_gene_factors,
    plot_eigenstate_factors,
)
from ..figures.commonFuncs.plotPaCMAP import plot_labels_pacmap
from ..figures.commonFuncs.plotGeneral import bal_combine_bo_covid
import seaborn as sns
import numpy as np
import pandas as pd

from pf2.data_import import convert_to_patients, import_data, import_meta
from pf2.figures.common import getSetup
from pf2.predict import predict_mortality
from pf2.tensor import pf2


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)
    
    meta = import_meta()
    data = import_data()
    conversions = convert_to_patients(data)
    
    # ranks = np.arange(5, 65, 5)
    ranks = np.arange(2, 4)
    r2xs = pd.Series(0, dtype=float, index=ranks)
    accuracies = pd.Series(0, dtype=float, index=ranks)
    for rank in ranks:
        fac, r2x = pf2(data, rank, do_embedding=False)
        patient_factor = pd.DataFrame(
            fac.uns["Pf2_A"],
            index=conversions,
            columns=np.arange(fac.uns["Pf2_A"].shape[1]) + 1,
        )
        if meta.shape[0] != patient_factor.shape[0]:
            meta = meta.loc[patient_factor.index, :]

        acc, _, _ = predict_mortality(patient_factor, meta)
        r2xs.loc[rank] = r2x
        accuracies.loc[rank] = acc
    
    ax[0].plot(ranks, r2xs)
    ax[0].set(xticks = ranks, ylabel = "R2X", xlabel = "Rank")
    ax[1].plot(ranks, accuracies,)
    ax[1].set(xticks = ranks, ylabel = "Accuracy", xlabel = "Rank")


    return f
