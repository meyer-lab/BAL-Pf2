import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold


SKF = StratifiedKFold(n_splits=5)


def predict_mortality(data, labels, proba=False):
    """Predicts mortality via cross-validation."""
    l1_ratios = np.linspace(0, 1, 11)
    model = LogisticRegressionCV(
        l1_ratios=l1_ratios,
        solver="saga",
        penalty="elasticnet",
        n_jobs=1,
        cv=SKF,
        max_iter=100000,
        multi_class='ovr'
    )
    model.fit(data, labels)

    if proba:
        probabilities = pd.Series(0, dtype=float, index=data.index)
        lr_model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=model.l1_ratio_[0],
            C=model.C_[0],
            max_iter=100000
        )
        for train_index, test_index in SKF.split(data, labels):
            train_data, train_labels = data.iloc[train_index, :], labels.iloc[train_index]
            test_data = data.iloc[test_index]
            lr_model.fit(train_data, train_labels)
            predicted = lr_model.predict_proba(test_data)
            probabilities.loc[test_data.index] = predicted[:, 1]
        return probabilities
    else:
        scores = np.mean(list(model.scores_.values())[0], axis=0)
        return np.max(scores), model.coef_[0]


def main():
    patients = pd.read_csv(
        f'output/10_fold_unscaled/patient_factors_rank_2.csv',
        index_col=0
    ).index
    conversions = pd.read_csv(
        f'data/SCRIPT_pt_bal_anonymized_ids_external.csv',
        index_col=0
    )

    in_conversions = patients.isin(conversions.index)
    conversions = conversions.loc[
        patients[in_conversions],
        'External patient ID'
    ]

    meta = pd.read_csv(
        f'data/Carpediem/CarpeDiem_dataset.csv',
        index_col=0
    )
    meta = meta.set_index('Patient_id', drop=True)
    meta = meta.loc[~meta.index.duplicated(), :]
    in_meta = conversions.isin(meta.index)

    labels = meta.loc[
        conversions.loc[in_meta],
        'Binary_outcome'
    ]

    data = pd.read_csv(
        'Patient_factor.csv',
        index_col=0
    )
    data = data.loc[in_conversions, :].loc[in_meta, :]
    data = data / data.max(axis=0)

    score, model = predict_mortality(data, labels)
    print(score)


if __name__ == '__main__':
    main()
