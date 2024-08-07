from typing import Tuple

import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale

SKF = StratifiedKFold(n_splits=10)


def run_plsr(
    data: pd.DataFrame,
    labels: pd.Series,
    proba: bool = False,
    n_components: int = 2
) -> tuple[pd.Series, PLSRegression]:
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

    data[:] = scale(data)
    plsr = PLSRegression(
        n_components=n_components,
        scale=False,
        max_iter=int(1E5)
    )
    rfe_cv = RFECV(plsr, step=1, cv=SKF, min_features_to_select=n_components)
    rfe_cv.fit(data, labels)
    data = data.loc[:, rfe_cv.support_]

    probabilities = pd.Series(0, dtype=float, index=data.index)
    for train_index, test_index in SKF.split(data, labels):
        train_group_data = data.iloc[train_index, :]
        train_labels = labels.iloc[train_index]
        test_group_data = data.iloc[test_index]
        plsr.fit(train_group_data, train_labels)
        probabilities.iloc[test_index] = plsr.predict(test_group_data)

    plsr.fit(data, labels)
    plsr.coef_ = pd.Series(
        plsr.coef_.squeeze(),
        index=data.columns
    )

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
