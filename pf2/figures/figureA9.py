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
    ax, f = getSetup((4, 4), (1, 1))
    subplotLabel(ax)
    
    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted_uncorrected.h5ad")

    # X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    
    # print(np.ravel(X.uns["Pf2_A"]))
    # print(np.ravel(XX.uns["Pf2_A"]))
    
    # assert np.ravel(X.uns["Pf2_A"]).all() != np.ravel(XX.uns["Pf2_A"]).all()
    
    meta = import_meta()
    
    conversions = convert_to_patients(X)


    patient_factor = pd.DataFrame(
        X.uns["Pf2_A"],
        index=conversions,
        columns=np.arange(X.uns["Pf2_A"].shape[1]) + 1,
    )
    meta = meta.loc[patient_factor.index, :]


    component_counts = np.arange(1, 11)
    print(component_counts)
    accuracies = pd.DataFrame(
        index=component_counts,
        columns=["Overall", "C19", "nC19"]
    )

    for n_components in component_counts:
        probabilities, labels = predict_mortality(
            patient_factor,
            meta,
            n_components=n_components,
            proba=True
        )
        probabilities = probabilities.round().astype(int)
        _meta = meta.loc[~meta.index.duplicated()].loc[labels.index]

        covid_acc = accuracy_score(
            labels.loc[_meta.loc[:, "patient_category"] == "COVID-19"],
            probabilities.loc[_meta.loc[:, "patient_category"] == "COVID-19"]
        )
        nc_acc = accuracy_score(
            labels.loc[_meta.loc[:, "patient_category"] != "COVID-19"],
            probabilities.loc[_meta.loc[:, "patient_category"] != "COVID-19"]
        )
        acc = accuracy_score(labels, probabilities)

        accuracies.loc[
            n_components,
            :
        ] = [acc, covid_acc, nc_acc]

    for column in accuracies.columns:
        if column == "Overall":
            ax[0].plot(
                component_counts,
                accuracies.loc[:, column],
                label=column,
                color="k",
                linestyle="--"
            )
        else:
            ax[0].plot(
                component_counts,
                accuracies.loc[:, column],
                label=column
            )

    ax[0].set_ylim([0, 1])

    ax[0].legend()
    ax[0].set_ylabel("Accuracy")
    ax[0].set_xlabel("PLSR Components")
    ax[0].set(xticks=np.arange(1, 11))





    return f
