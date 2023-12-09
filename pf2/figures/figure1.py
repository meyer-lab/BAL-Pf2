"""Figure 1: Factor Heatmaps"""
import numpy as np
import pandas as pd
import seaborn as sns

from pf2.data_import import import_data
from pf2.figures.common import getSetup
from pf2.tensor import pf2


def makeFigure():
    data = import_data()
    data, _ = pf2(data, do_embedding=False)

    factors = {}
    dims = ["Patient", "Cell State", "Gene"]
    for factor, dim in zip(
        [data.uns["Pf2_A"], data.uns["Pf2_B"], data.varm["Pf2_C"]], dims
    ):
        factors[dim] = pd.DataFrame(
            factor,
            columns=np.arange(data.uns["Pf2_rank"]) + 1,
        )

    axs, fig = getSetup((8, 4), (1, len(factors)))
    for ax, dim in zip(axs, factors.keys()):
        factor = factors[dim]
        sns.heatmap(
            factor, vmin=-1, vmax=1, cmap="coolwarm", cbar=ax == axs[-1], ax=ax
        )
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(dim)

    return fig
