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
from typing import Tuple

import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale
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
from ..data_import import condition_factors_meta
# from pf2.predict import predict_mortality, run_plsr

SKF = StratifiedKFold(n_splits=10)

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((5, 5), (1, 1))
    subplotLabel(ax)
    
    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    # X.uns["Pf2_A"] = correct_conditions(X)

    meta = import_meta()
    conversions = convert_to_patients(X)

    patient_factor = pd.DataFrame(
        X.uns["Pf2_A"],
        index=conversions,
        columns=np.arange(X.uns["Pf2_A"].shape[1]) + 1,
    )
    meta = meta.loc[patient_factor.index, :]
    

    # print(meta.columns)
    # for i in range(len(meta.columns)):
    #     # print(meta.columns[i])
    #     print(meta.dtypes.iloc[i])

    # print(meta.dtypes)
    # print(meta)

    patient_factor = pd.concat([patient_factor, meta], axis=1)

    patient_factorss = patient_factor.loc[:, patient_factor.dtypes == np.float64]
    # patient_factorss = pd.concat([patient_factor.loc[:, patient_factor.dtypes == np.float64], patient_factor.loc[:, patient_factor.dtypes == np.int64]], axis=1)
    patient_factorss["patient_category"] = patient_factor["patient_category"] 
    patient_factorss["binary_outcome"] = patient_factor["binary_outcome"] 
    patient_factorss.columns = patient_factorss.columns.astype(str)
    
    # print(patient_factor)
    patient_factorss.dropna(thresh=0.9*len(patient_factorss), axis=1, inplace=True)
    # print(patient_factor)
    patient_factorss.dropna(inplace=True)
    
    meta = patient_factorss[["patient_category", "binary_outcome"]]
    
    patient_factorss = patient_factorss.drop(columns=["patient_category", "binary_outcome"])
    
    accuracies = pd.DataFrame(
        columns=["Overall", "C19", "nC19"]
    )

    # patient_factorss = patient_factorss.iloc[:, :50]
    patient_factorss = patient_factorss.iloc[:, 50:]
    print(patient_factorss)
    probabilities, labels = predict_mortality(
        patient_factorss,
        meta,
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
        0,
        :
        ] = [acc, covid_acc, nc_acc]
        
    print(accuracies)



    sns.barplot(data=accuracies, ax=ax[0])

    ax[0].set_ylim([0, 1])
    ax[0].legend()
    ax[0].set_ylabel("Accuracy")


    
    
    

    return f

def run_plsr(
    data: pd.DataFrame,
    labels: pd.Series,
    proba: bool = False,
    n_components: int = 2
):
    """
    Predicts labels via PLSR cross-validation.

    Args:
        data (pd.DataFrame): data to predict
        labels (pd.Series): classification labels
        proba (bool, default:False): return probability of prediction

    Returns:
        predicted (pd.Series): predicted mortality for patients; if proba is
            True, returns probabilities of mortality
        plsr (PLSRegression): fitted PLSR model
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    # data[:] = scale(data)
    plsr = PLSRegression(
        n_components=n_components,
        scale=False,
        max_iter=int(1E5))
    
    
        

    probabilities = pd.Series(0, dtype=float, index=data.index)
    for train_index, test_index in SKF.split(data, labels):
        train_group_data = data.iloc[train_index, :]
        train_labels = labels.iloc[train_index]
        test_group_data = data.iloc[test_index]
        plsr.fit(train_group_data, train_labels)
        probabilities.iloc[test_index] = plsr.predict(test_group_data)

    plsr.fit(data, labels)
    
    if proba:
        return probabilities, plsr
    else:
        predicted = probabilities.round().astype(int)
        return predicted, plsr




def predict_mortality(
    data: pd.DataFrame,
    meta: pd.DataFrame,
    proba: bool = False,
    n_components = 2
):
    """
    Predicts mortality via cross-validation.

    Parameters:
        data (pd.DataFrame): data to predict
        meta (pd.DataFrame): patient meta-data
        proba (bool, default:False): return probability of prediction

    Returns:
        if proba:
            probabilities (pd.Series): predicted probability of mortality for
                patients
            labels (pd.Series): classification targets
        else:
            accuracy (float): prediction accuracy
            models (tuple[COVID, Non-COVID]): fitted PLSR models
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    data = data.loc[
        meta.loc[:, "patient_category"] != "Non-Pneumonia Control",
        :
    ]
    meta = meta.loc[
        meta.loc[:, "patient_category"] != "Non-Pneumonia Control",
        :
    ]
    labels = data.index.to_series().replace(meta.loc[:, "binary_outcome"])

    covid_data = data.loc[meta.loc[:, "patient_category"] == "COVID-19", :]
    covid_labels = meta.loc[
        meta.loc[:, "patient_category"] == "COVID-19",
        "binary_outcome"
    ]
    nc_data = data.loc[meta.loc[:, "patient_category"] != "COVID-19", :]
    nc_labels = meta.loc[
        meta.loc[:, "patient_category"] != "COVID-19",
        "binary_outcome"
    ]

    predictions = pd.Series(index=data.index)
    predictions.loc[meta.loc[:, "patient_category"] == "COVID-19"], c_plsr = \
        run_plsr(
            covid_data, covid_labels, proba=proba, n_components=n_components
        )
    predictions.loc[meta.loc[:, "patient_category"] != "COVID-19"], nc_plsr = \
        run_plsr(
            nc_data, nc_labels, proba=proba, n_components=n_components
        )

    if proba:
        return predictions, labels
    else:
        predicted = predictions.round().astype(int)
        return accuracy_score(labels, predicted), (c_plsr, nc_plsr)