import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
import anndata

SGKF = StratifiedGroupKFold(n_splits=10)

def logistic_regression(scoring):
    """Standardizing LogReg for all functions"""
    return LogisticRegressionCV(
        random_state=0,
        max_iter=10000,
        penalty="l1",
        solver="saga",
        scoring=scoring,
    )

def run_lr(
    data: pd.DataFrame,
    labels: pd.DataFrame,
    proba: bool = False,
    scoring: str = 'accuracy'
) -> tuple[pd.Series, LogisticRegressionCV]:
    """
    Predicts labels via logistic regression cross-validation.

    Args:
        data (pd.DataFrame): data to predict
        labels (pd.DataFrame): classification labels
        proba (bool, default:False): return probability of prediction
        scoring (str, default:'accuracy'): scoring metric for LogisticRegressionCV

    Returns:
        predicted (pd.Series): predicted mortality for patients; if proba is
            True, returns probabilities of mortality
        lr (LogisticRegressionCV): fitted logistic regression model
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    lr = logistic_regression(scoring)

    probabilities = pd.Series(0, dtype=float, index=data.index)
    for train_index, test_index in SGKF.split(
        data,
        labels.loc[:, "binary_outcome"],
        labels.loc[:, "patient_id"]
    ):
        train_group_data = data.iloc[train_index, :]
        train_labels = labels.iloc[train_index].loc[:, "binary_outcome"]
        test_group_data = data.iloc[test_index]
        lr.fit(train_group_data, train_labels)
        if proba:
            probabilities.iloc[test_index] = lr.predict_proba(test_group_data)[:, 1]
        else:
            probabilities.iloc[test_index] = lr.predict(test_group_data)

    lr.fit(data, labels.loc[:, "binary_outcome"])
    lr.coef_ = pd.Series(lr.coef_.squeeze(), index=data.columns)

    return probabilities, lr

    
def predict_mortality_all(
    X: anndata.AnnData, 
    data: pd.DataFrame, 
    proba: bool = False, 
    scoring: str = 'accuracy',
    bulk: bool = False
):
    """
    Predicts mortality via cross-validation without breaking up by status.

    Parameters:
        data (pd.DataFrame): data to predict
        meta (pd.DataFrame): patient meta-data
        proba (bool, default:False): return probability of prediction
        scoring (str, default:'accuracy'): scoring metric for LogisticRegressionCV
        bulk (bool, default:False): whether to use all features or just components

    Returns:
        if proba:
            probabilities (pd.Series): predicted probability of mortality for
                patients
            labels (pd.Series): classification targets
        else:
            accuracy (float): prediction accuracy
            labels (pd.Series): classification targets
            model: fitted logistic regression models
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    cond_fact_meta_df = data[data["patient_category"] != "Non-Pneumonia Control"]
    
    labels = cond_fact_meta_df.loc[:, ["binary_outcome", "patient_id"]]
    predictions = pd.Series(index=cond_fact_meta_df.index)
    
    if bulk is False:
        predictions[:], all_lr = run_lr(
            cond_fact_meta_df[[f"Cmp. {i}" for i in np.arange(1, X.uns["Pf2_A"].shape[1] + 1)]], 
            labels, proba=proba, scoring=scoring
        )
    else:
        predictions[:], all_lr = run_lr(
            cond_fact_meta_df.iloc[:, :-3],
            labels, proba=proba, scoring=scoring
        )
        
    if proba:
        return predictions, labels.loc[:, "binary_outcome"]
    else:
        return (
            accuracy_score(labels.loc[:, "binary_outcome"], predictions),
            labels.loc[:, "binary_outcome"],
            all_lr
        )
    

def predict_mortality(
    X: anndata.AnnData,
    data: pd.DataFrame,
    scoring: str = 'accuracy',
    proba: bool = False
) -> tuple[pd.Series, pd.Series, tuple[LogisticRegressionCV, LogisticRegressionCV]]:
    """
    Predicts mortality via cross-validation.

    Parameters:
        X (anndata.AnnData): factorization results
        data (pd.DataFrame): patient meta-data
        scoring (str, default:'accuracy'): scoring metric for LogisticRegressionCV
        proba (bool, default:False): return probability of prediction

    Returns:
        predictions (pd.Series): if proba, probabilities of mortality for each
            sample; else, predicted mortality outcome
        labels (pd.Series): classification targets
        models (tuple[COVID, Non-COVID]): fitted logistic regression models
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    cond_fact_meta_df = data.loc[
        data.loc[:, "patient_category"] != "Non-Pneumonia Control",
        :
    ]
    
    labels = cond_fact_meta_df.loc[
         :,
         ["binary_outcome", "patient_id"]
     ].astype(int)

    covid_data = cond_fact_meta_df.loc[cond_fact_meta_df.loc[:, "patient_category"] == "COVID-19", :]
    covid_labels = labels.loc[
        cond_fact_meta_df.loc[:, "patient_category"] == "COVID-19",
        :
    ]
    nc_data = cond_fact_meta_df.loc[cond_fact_meta_df.loc[:, "patient_category"] != "COVID-19", :]
    nc_labels = labels.loc[
        cond_fact_meta_df.loc[:, "patient_category"] != "COVID-19",
        :
    ]

    predictions = pd.Series(index=cond_fact_meta_df.index)
    predictions.loc[cond_fact_meta_df.loc[:, "patient_category"] == "COVID-19"], c_lr = run_lr(
        covid_data[[f"Cmp. {i}" for i in np.arange(1, X.uns["Pf2_A"].shape[1] + 1)]],
        covid_labels, proba=proba, scoring=scoring
    )
    predictions.loc[cond_fact_meta_df.loc[:, "patient_category"] != "COVID-19"], nc_lr = run_lr(
        nc_data[[f"Cmp. {i}" for i in np.arange(1, X.uns["Pf2_A"].shape[1] + 1)]],
        nc_labels, proba=proba, scoring=scoring
    )

    return (
        predictions,
        labels.loc[:, "binary_outcome"].squeeze(),
        (c_lr, nc_lr)
    )


def lr_acc_proba(X, patient_factor_matrix, scoring='accuracy', roc_auc=True):
    """Runs logistic regression and obtains average prediction accuracy"""

    acc_df = pd.DataFrame(columns=["Overall", "C19", "nC19"])

    probabilities_all, labels_all = predict_mortality_all(
        X, patient_factor_matrix, proba=True, scoring=scoring
    )

    if roc_auc:
        score = roc_auc_score
    else:
        score = accuracy_score
        
    covid_acc = score(
        labels_all.loc[patient_factor_matrix.loc[:, "patient_category"] == "COVID-19"].to_numpy().astype(int),
        probabilities_all.loc[patient_factor_matrix.loc[:, "patient_category"] == "COVID-19"],
    )
    nc_acc = score(
        labels_all.loc[patient_factor_matrix.loc[:, "patient_category"] != "COVID-19"].to_numpy().astype(int),
        probabilities_all.loc[patient_factor_matrix.loc[:, "patient_category"] != "COVID-19"],
    )
    acc = score(labels_all.to_numpy().astype(int), probabilities_all)

    acc_df.loc[0, :] = [acc, covid_acc, nc_acc]

    return acc_df


def lr_acc(X, patient_factor_matrix, scoring='accuracy'):
    """Runs logistic regression and obtains average prediction accuracy for C19 and nC19"""

    _, labels, [c19_lr, nc19_lr] = predict_mortality(X,
        patient_factor_matrix, scoring=scoring
    )

    return labels, [c19_lr, nc19_lr]