"""
Figure A8:
"""

import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.axes import Axes
import anndata
from .common import subplotLabel, getSetup
from ..tensor import correct_conditions
from .figureA4 import partial_correlation_matrix
import numpy as np
import pandas as pd
from anndata import read_h5ad
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from pf2.data_import import convert_to_patients, import_meta
from pf2.figures.common import getSetup
from pf2.predict import predict_mortality, run_plsr


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((8, 8), (2, 2))
    subplotLabel(ax)
    
    meta = import_meta()
    data = read_h5ad("/opt/northwest_bal/full_fitted_uncorrected.h5ad", backed="r")
    conversions = convert_to_patients(data)

    patient_factor = pd.DataFrame(
        data.uns["Pf2_A"],
        index=conversions,
        columns=np.arange(data.uns["Pf2_A"].shape[1]) + 1,
    )
    meta = meta.loc[patient_factor.index, :]

    accuracies, (covid_plsr, nc_plsr) = predict_mortality(
        patient_factor,
        meta
    )


    names = ["COVID-19", "Non-COVID 19"]
    plsr_methods = [covid_plsr, nc_plsr]
    for ax_index, (name, plsr) in enumerate(zip(names, plsr_methods)):
        x_ax = ax[ax_index]
        print(plsr)
        x_ax.scatter(
            plsr.y_loadings_[0, 0],
            plsr.y_loadings_[0, 1],
            s=150,
            c="tab:red"
        )
        x_ax.scatter(
            plsr.x_loadings_[:, 0],
            plsr.x_loadings_[:, 1],
            s=120,
            facecolors="white",
            edgecolors="k"
        )
        for index, component in enumerate(plsr.coef_.index):
            x_ax.text(
                plsr.x_loadings_[index, 0],
                plsr.x_loadings_[index, 1] - 0.001,
                ha="center",
                ma="center",
                va="center",
                s=component
            )

        x_lims = x_ax.get_xlim()
        y_lims = x_ax.get_ylim()

        x_ax.plot([-100, 100], [0, 0], linestyle="--", color="k", zorder=-3)
        x_ax.plot([0, 0], [-100, 100], linestyle="--", color="k", zorder=-3)
        x_ax.set_xlim(x_lims)
        x_ax.set_ylim(y_lims)

        x_ax.set_xlabel("PLSR 1")
        x_ax.set_ylabel("PLSR 2")
        x_ax.set_title(f"{name}: X-loadings")


    return f
