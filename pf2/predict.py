import numpy as np
import pandas as pd
from sklearn.linear_model import (LinearRegression, LogisticRegression,
                                  LogisticRegressionCV)
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import scale

KF = KFold(n_splits=5)
SKF = StratifiedKFold(n_splits=5)


def predict_mortality(data, labels, proba=False, pred=False):
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
    l1_ratios = np.linspace(0, 1, 11)
    labels = labels.astype(int)
    model = LogisticRegressionCV(
        l1_ratios=l1_ratios,
        solver="saga",
        penalty="elasticnet",
        n_jobs=1,
        cv=SKF,
        scoring="balanced_accuracy",
        max_iter=100000,
        multi_class="ovr"
    )
    model.fit(data, labels)

    if proba:
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
        return probabilities
    elif pred:
        predicted = pd.Series(0, dtype=float, index=data.index)
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
            predicted.iloc[test_index] = lr_model.predict(test_data)
        return predicted
    else:
        scores = np.mean(list(model.scores_.values())[0], axis=0)
        return np.max(scores), model.coef_[0]


def predict_regression(data, labels):
    """
    Predicts mortality via cross-validation.

    Parameters:
        data (pd.DataFrame): data to predict
        labels (pd.Series): labels to predict from data

    Returns:
        if proba:
            probabilities (pd.Series): probability of mortality for patients;
                shares index with labels
        else:
            accuracy (float): prediction accuracy
            coefficients (np.array): LR model coefficients
    """
    # ols = LinearRegression()
    model = SVR()
    predicted = pd.Series(0, dtype=float, index=data.index)

    for train_index, test_index in KF.split(data, labels):
        train_data = data.iloc[train_index, :]
        train_labels = labels.iloc[train_index]
        test_data = data.iloc[test_index]
        model.fit(scale(train_data), train_labels)
        predicted.iloc[test_index] = model.predict(scale(test_data))

    score = r2_score(labels, predicted)
    return score, predicted
