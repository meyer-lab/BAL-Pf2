import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale

SKF = StratifiedKFold(n_splits=10)


def predict_mortality(data, labels, proba=False):
    """
    Predicts mortality via cross-validation.

    Parameters:
        data (pd.DataFrame): data to predict
        labels (pd.Series): labels to predict from data
        proba (bool, default:False): return probability of prediction

    Returns:
        if proba:
            probabilities (pd.Series): probability of mortality for patients;
                shares index with labels
        else:
            accuracy (float): prediction accuracy
            coefficients (np.array): LR model coefficients
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    data[:] = scale(data)
    l1_ratios = np.linspace(0, 1, 11)
    rfe_model = LogisticRegression()

    rfe_cv = RFECV(rfe_model, step=1, cv=SKF)
    rfe_cv.fit(data, labels)
    data = data.loc[:, rfe_cv.support_]

    model = LogisticRegressionCV(
        Cs=11,
        l1_ratios=l1_ratios,
        solver="saga",
        penalty="elasticnet",
        n_jobs=1,
        cv=SKF,
        max_iter=100000
    )
    model.fit(data, labels)

    probabilities = pd.Series(0, dtype=float, index=data.index)
    lr_model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=model.l1_ratio_[0],
        C=model.C_[0],
        max_iter=100000,
    )
    for train_index, test_index in SKF.split(data, labels):
        train_data = data.iloc[train_index, :]
        train_labels = labels.iloc[train_index]
        test_data = data.iloc[test_index]
        lr_model.fit(train_data, train_labels)
        predicted = lr_model.predict_proba(test_data)
        probabilities.iloc[test_index] = predicted[:, 1]

    lr_model.fit(data, labels)

    if proba:
        return probabilities
    else:
        predicted = probabilities.round().astype(int)
        coefficients = pd.Series(
            lr_model.coef_[0],
            index=np.argwhere(rfe_cv.support_).flatten() + 1
        )
        return accuracy_score(labels, predicted), coefficients
