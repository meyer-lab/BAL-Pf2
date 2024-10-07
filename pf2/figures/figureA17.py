"""
Figure A17:
"""

import pandas as pd
from ..figures.common import getSetup, subplotLabel
import numpy as np
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
    
    # ranks = np.arange(5, 85, 5)
    # ranks = np.arange(2, 4)
    # r2xs = pd.Series(0, dtype=float, index=ranks)
    # accuracies = pd.Series(0, dtype=float, index=ranks)
    # for rank in ranks:
    #     fac, r2x = pf2(data, rank, do_embedding=False)
    #     patient_factor = pd.DataFrame(
    #         fac.uns["Pf2_A"],
    #         index=conversions,
    #         columns=np.arange(fac.uns["Pf2_A"].shape[1]) + 1,
    #     )
    #     if meta.shape[0] != patient_factor.shape[0]:
    #         meta = meta.loc[patient_factor.index, :]

    #     acc, _, _ = predict_mortality(patient_factor, meta)
    #     r2xs.loc[rank] = r2x
    #     accuracies.loc[rank] = acc
    
    # ax[0].plot(ranks, r2xs, color = "k")
    # ax[0].set(xticks = ranks, ylabel = "R2X", xlabel = "Rank")
    # ax[0].set_ylim(bottom=0)
    # ax[1].plot(ranks, accuracies, color = "k")
    # ax[1].set(ylim=[0, 1], xticks = ranks, ylabel = "Accuracy", xlabel = "Rank")

    


    return f