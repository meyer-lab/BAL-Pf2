"""Figure 1: Factor Heatmaps"""

import numpy as np
import pandas as pd
import seaborn as sns
from anndata import read_h5ad

from pf2.figures.common import getSetup


def makeFigure():
    data = read_h5ad("/opt/northwest_bal/full_fitted.h5ad", backed="r")

    factors = {}
    dims = ["Patient", "Cell State"]
    for factor, dim in zip([data.uns["Pf2_A"], data.uns["Pf2_B"]], dims):
        factors[dim] = pd.DataFrame(
            factor,
            columns=np.arange(data.uns["Pf2_A"].shape[1]) + 1,
        )

    axs, fig = getSetup((8, 4), (1, len(factors)))
    for ax, dim in zip(axs, factors.keys()):
        factor = factors[dim]
        sns.heatmap(factor, vmin=-1, vmax=1, cmap="coolwarm", cbar=ax == axs[-1], ax=ax)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(dim)

    return fig
