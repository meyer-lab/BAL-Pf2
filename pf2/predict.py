import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale

SKF = StratifiedKFold(n_splits=10)


def run_lr(
    data: pd.DataFrame, labels: pd.Series, proba: bool = False
) -> tuple[pd.Series, pd.Series]:
    """
    Predicts labels via logistic regression cross-validation.

    Args:
        data (pd.DataFrame): data to predict
        labels (pd.Series): classification labels
        proba (bool, default:False): return probability of prediction

    Returns:
        if proba:
            probabilities (pd.Series): predicted probability of mortality for
                patients; shares index with labels
            coefficients (pd.Series): LR model coefficients
        else:
            probabilities (pd.Series): predicted mortality outcome for patients;
                shares index with labels
            coefficients (pd.Series): LR model coefficients
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    data[:] = scale(data)
    lr_model = LogisticRegression()
    coefficients = pd.Series(0, index=data.columns, dtype=float)

    rfe_cv = RFECV(lr_model, step=1, cv=SKF)
    rfe_cv.fit(data, labels)
    data = data.loc[:, rfe_cv.support_]

    probabilities = pd.Series(0, dtype=float, index=data.index)
    for train_index, test_index in SKF.split(data, labels):
        train_group_data = data.iloc[train_index, :]
        train_labels = labels.iloc[train_index]
        test_group_data = data.iloc[test_index]
        lr_model.fit(train_group_data, train_labels)
        predicted = lr_model.predict_proba(test_group_data)
        probabilities.iloc[test_index] = predicted[:, 1]

    lr_model.fit(data, labels)
    coefficients.loc[rfe_cv.support_] = lr_model.coef_.squeeze()

    if proba:
        return probabilities, coefficients
    else:
        predicted = probabilities.round().astype(int)
        return predicted, coefficients


def predict_mortality(data: pd.DataFrame, meta: pd.DataFrame, proba: bool = False):
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
            coefficients (pd.DataFrame): LR model coefficients; columns
                correspond to unique models for COVID/non-COVID patients
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    data = data.loc[meta.loc[:, "patient_category"] != "Non-Pneumonia Control", :]
    meta = meta.loc[meta.loc[:, "patient_category"] != "Non-Pneumonia Control", :]
    labels = data.index.to_series().replace(meta.loc[:, "binary_outcome"])
    groups = meta.loc[:, "patient_category"] == "COVID-19"
    groups = groups.astype(int)

    predictions = pd.Series(index=data.index)
    coefficients = pd.DataFrame(index=data.columns, columns=np.unique(groups))

    for group in groups.unique():
        group_data = data.loc[groups == group, :]
        group_labels = labels.loc[groups == group]

        group_predictions, group_coefficients = run_lr(
            group_data, group_labels, proba=proba
        )
        predictions.loc[groups == group] = group_predictions
        coefficients.loc[:, group] = group_coefficients

    coefficients.columns = ["Non-COVID", "COVID-19"]

    if proba:
        return predictions, labels
    else:
        predicted = predictions.round().astype(int)
        return accuracy_score(labels, predicted), coefficients
