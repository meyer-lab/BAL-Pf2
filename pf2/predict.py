import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
import anndata
SKF = StratifiedKFold(n_splits=10)


def run_plsr(
    data: pd.DataFrame, labels: pd.Series, proba: bool = False, n_components: int = 1
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

    plsr = PLSRegression(n_components=n_components, scale=True, max_iter=int(1e5))

    probabilities = pd.Series(0, dtype=float, index=data.index)
    for train_index, test_index in SKF.split(data, labels):
        train_group_data = data.iloc[train_index, :]
        train_labels = labels.iloc[train_index]
        test_group_data = data.iloc[test_index]
        plsr.fit(train_group_data, train_labels)
        probabilities.iloc[test_index] = plsr.predict(test_group_data)

    plsr.fit(data, labels)
    plsr.coef_ = pd.Series(plsr.coef_.squeeze(), index=data.columns)

    if proba:
        return probabilities, plsr

    else:
        predicted = probabilities.round().astype(int)
        return predicted, plsr

    
def predict_mortality_all(
    X: anndata.AnnData, data: pd.DataFrame, proba: bool = False, n_components=1, bulk=False
):
    """
    Predicts mortality via cross-validation without breaking up by status.

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
            labels (pd.Series): classification targets
            model: fitted PLSR models
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    cond_fact_meta_df = data[data["patient_category"] != "Non-Pneumonia Control"]
    
    labels = cond_fact_meta_df["binary_outcome"]
    labels = pd.Series(index=labels.index, data=labels.to_numpy().astype(int))
    predictions = pd.Series(index=cond_fact_meta_df.index)
    
    if bulk is False:
        predictions[:], all_plsr = run_plsr(
            cond_fact_meta_df[[f"Cmp. {i}" for i in np.arange(1, X.uns["Pf2_A"].shape[1] + 1)]], 
            labels, proba=proba, n_components=n_components
        )
    else:
        predictions[:], all_plsr = run_plsr(
            cond_fact_meta_df.iloc[:, :-2], 
            labels, proba=proba, n_components=n_components
        )
        
    if proba:
        return predictions, labels

    else:
        predicted = predictions.round().astype(int)
        return  accuracy_score(labels, predicted), labels, all_plsr
    

def predict_mortality(
    X: anndata.AnnData,
    data: pd.DataFrame,
    n_components: int = 1,
    proba: bool = False
) -> tuple[pd.Series, pd.Series, tuple[PLSRegression, PLSRegression]]:
    """
    Predicts mortality via cross-validation.

    Parameters:
        X (anndata.AnnData): factorization results
        data (pd.DataFrame): patient meta-data
        n_components (int, default:1): number of PLS components to use
        proba (bool, default:False): return probability of prediction

    Returns:
        predictions (pd.Series): if proba, probabilities of mortality for each
            sample; else, predicted mortality outcome
        labels (pd.Series): classification targets
        models (tuple[COVID, Non-COVID]): fitted PLSR models
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    cond_fact_meta_df = data[data["patient_category"] != "Non-Pneumonia Control"]
    
    labels = cond_fact_meta_df["binary_outcome"]
    labels = pd.Series(index=labels.index, data=labels.to_numpy().astype(int))

    covid_data = cond_fact_meta_df.loc[cond_fact_meta_df.loc[:, "patient_category"] == "COVID-19", :]
    covid_labels = cond_fact_meta_df.loc[
        cond_fact_meta_df.loc[:, "patient_category"] == "COVID-19", "binary_outcome"
    ]
    nc_data = cond_fact_meta_df.loc[cond_fact_meta_df.loc[:, "patient_category"] != "COVID-19", :]
    nc_labels = cond_fact_meta_df.loc[
        cond_fact_meta_df.loc[:, "patient_category"] != "COVID-19", "binary_outcome"
    ]

    predictions = pd.Series(index=cond_fact_meta_df.index)
    predictions.loc[cond_fact_meta_df.loc[:, "patient_category"] == "COVID-19"], c_plsr = run_plsr(
        covid_data[[f"Cmp. {i}" for i in np.arange(1, X.uns["Pf2_A"].shape[1] + 1)]],
        covid_labels, proba=proba, n_components=n_components
    )
    predictions.loc[cond_fact_meta_df.loc[:, "patient_category"] != "COVID-19"], nc_plsr = run_plsr(
        nc_data[[f"Cmp. {i}" for i in np.arange(1, X.uns["Pf2_A"].shape[1] + 1)]],
        nc_labels, proba=proba, n_components=n_components
    )

    return predictions, labels, (c_plsr, nc_plsr)


def plsr_acc_proba(X, patient_factor_matrix, n_components=1, roc_auc=True):
    """Runs PLSR and obtains average prediction accuracy"""

    acc_df = pd.DataFrame(columns=["Overall", "C19", "nC19"])

    probabilities_all, labels_all = predict_mortality_all(
        X, patient_factor_matrix, n_components=n_components, proba=True
    )

    if roc_auc:
        score = roc_auc_score
    else:
        score = accuracy_score
        
    covid_acc = score(
        labels_all.loc[patient_factor_matrix.loc[:, "patient_category"] == "COVID-19"].to_numpy().astype(int),
        probabilities_all.loc[patient_factor_matrix.loc[:, "patient_category"] == "COVID-19"].round().astype(int),
    )
    nc_acc = score(
        labels_all.loc[patient_factor_matrix.loc[:, "patient_category"] != "COVID-19"].to_numpy().astype(int),
        probabilities_all.loc[patient_factor_matrix.loc[:, "patient_category"] != "COVID-19"].round().astype(int),
    )
    acc = score(labels_all.to_numpy().astype(int), probabilities_all.round().astype(int))

    acc_df.loc[0, :] = [acc, covid_acc, nc_acc]

    return acc_df


def plsr_acc(X, patient_factor_matrix, n_components=1):
    """Runs PLSR and obtains average prediction accuracy for C19 and nC19"""

    _, labels, [c19_plsr, nc19_plsr] = predict_mortality(X,
        patient_factor_matrix, n_components=n_components,
    )

    return labels, [c19_plsr, nc19_plsr]
