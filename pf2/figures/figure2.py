"""Figure 2: R2X Curve"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from pf2.figures.common import getSetup
from pf2.tensor import build_tensor, import_data, run_parafac2


def makeFigure():
    data = import_data()
    tensor, patients = build_tensor(data)
    ranks = np.arange(1, 41)
    r2x = pd.Series(0, dtype=float, index=ranks)
    for rank in tqdm(ranks):
        _, r2x = run_parafac2(tensor, rank)
        r2x.loc[rank] = r2x

    axs, fig = getSetup(
        (8, 4),
        (1, 1)
    )
    ax = axs[0]

    ax.plot(r2x.index, r2x)
    ax.grid(True)

    ax.set_ylabel("R2X")
    ax.set_xlabel("Rank")

    return fig
