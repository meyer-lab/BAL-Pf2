from os.path import abspath, dirname

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pf2.figures.common import getSetup
from pf2.data_import import import_data
from pf2.tensor import build_tensor, run_parafac2


def makeFigure():
    data = import_data()
    tensor, patients = build_tensor(data)
    pf2 = run_parafac2(tensor)

    factors = {}
    dims = ["Patient", "Cell State", "Gene"]
    for factor, dim in zip(pf2.factors, dims):
        factors[dim] = pd.DataFrame(
            factor / abs(factor).max(axis=0), columns=np.arange(pf2.rank) + 1
        )

    axs, fig = getSetup(
        (8, 4),
        (1, len(factors))
    )
    for ax, dim in zip(axs, factors.keys()):
        factor = factors[dim]
        sns.heatmap(
            factor,
            vmin=-1,
            vmax=1,
            cmap="coolwarm",
            cbar=ax == axs[-1],
            ax=ax
        )
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(dim)

    return fig
