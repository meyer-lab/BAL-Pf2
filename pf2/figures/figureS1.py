"""
Figure A17:
"""

import pandas as pd
from .common import getSetup, subplotLabel
from ..tensor import correct_conditions
import numpy as np
from pf2.data_import import convert_to_patients, import_data, import_meta
from pf2.figures.common import getSetup
from pf2.predict import predict_mortality
from pf2.tensor import pf2


def makeFigure():
    ax, f = getSetup((6, 6), (2, 2))
    subplotLabel(ax)
    
    X = import_data()
    meta = import_meta(drop_duplicates=False)
    conversions = convert_to_patients(X, sample=True)
    meta.set_index("sample_id", inplace=True)
    # ranks = np.arange(5, 65, 5)
    ranks = np.arange(2, 3)
    r2xs = pd.Series(0, dtype=float, index=ranks)
    accuracies = pd.Series(0, dtype=float, index=ranks)
    for rank in ranks:
        fac, r2x = pf2(X, rank, do_embedding=False)
        fac.uns["Pf2_A"] = correct_conditions(X)
        patient_factor = pd.DataFrame(
            fac.uns["Pf2_A"],
            index=conversions,
            columns=np.arange(fac.uns["Pf2_A"].shape[1]) + 1,
        )
        shared_indices = patient_factor.index.intersection(meta.index)
        patient_factor = patient_factor.loc[shared_indices, :]
        meta = meta.loc[shared_indices, :]

        acc, _, _ = predict_mortality(patient_factor, meta)
        r2xs.loc[rank] = r2x
        accuracies.loc[rank] = acc
    
    ax[0].plot(ranks, r2xs)
    ax[0].set(xticks = ranks, ylabel = "R2X", xlabel = "Rank")
    ax[1].plot(ranks, accuracies,)
    ax[1].set(xticks = ranks, ylabel = "Accuracy", xlabel = "Rank")


    return f