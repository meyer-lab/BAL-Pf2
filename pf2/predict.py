import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score
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
    X: anndata.AnnData, data: pd.DataFrame, proba: bool = False, n_components=1
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
    predictions[:], all_plsr = run_plsr(
        cond_fact_meta_df[[f"Cmp. {i}" for i in np.arange(1, X.uns["Pf2_A"].shape[1] + 1)]], 
        labels, proba=proba, n_components=n_components
    )

    if proba:
        return predictions, labels

    else:
        predicted = predictions.round().astype(int)
        return  accuracy_score(labels, predicted), labels, all_plsr
    

def predict_mortality(
    X, data: pd.DataFrame, n_components=1
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

    
    cond_fact_meta_df = data[data["patient_category"] != "Non-Pneumonia Control"]
    
    labels = cond_fact_meta_df["binary_outcome"]
    labels = pd.Series(index=labels.index, data=labels.to_numpy().astype(int))
    predictions = pd.Series(index=cond_fact_meta_df.index)

    covid_data = cond_fact_meta_df.loc[cond_fact_meta_df.loc[:, "patient_category"] == "COVID-19", :]
    covid_labels = cond_fact_meta_df.loc[
        cond_fact_meta_df.loc[:, "patient_category"] == "COVID-19", "binary_outcome"
    ]
    nc_data = cond_fact_meta_df.loc[cond_fact_meta_df.loc[:, "patient_category"] != "COVID-19", :]
    nc_labels = cond_fact_meta_df.loc[
        cond_fact_meta_df.loc[:, "patient_category"] != "COVID-19", "binary_outcome"
    ]

    predictions = pd.Series(index=data.index)
    predictions.loc[cond_fact_meta_df.loc[:, "patient_category"] == "COVID-19"], c_plsr = run_plsr(
          covid_data[[f"Cmp. {i}" for i in np.arange(1, X.uns["Pf2_A"].shape[1] + 1)]],
          covid_labels, proba=False, n_components=n_components
    )
    predictions.loc[cond_fact_meta_df.loc[:, "patient_category"] != "COVID-19"], nc_plsr = run_plsr(
        nc_data[[f"Cmp. {i}" for i in np.arange(1, X.uns["Pf2_A"].shape[1] + 1)]],
        nc_labels, proba=False, n_components=n_components
    )

    return  labels, (c_plsr, nc_plsr)