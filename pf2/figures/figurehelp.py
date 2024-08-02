"""Figure BS_S3: Logistic Regression on patient characteristics"""

import anndata
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
from .common import subplotLabel, getSetup
from ..data_import import condition_factors_meta
from ..tensor import correct_conditions

#logistic regression tools
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.preprocessing import scale


def makeFigure():
    ax, f = getSetup((10,12), (2,2))

    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    X.uns["Pf2_A"] = correct_conditions(X)
    cond_factors_df = condition_factors_meta(X)
    
    feature = "covid_status"
    #feature_2 ='smoking_status'
    #feature_3 = 'patient_category'
    #target = 'binary_outcome'
    
    #dropping Nan values
    cond_factors_df = cond_factors_df.dropna(subset = feature)
    
    factors_only = [f"Cmp. {i}" for i in np.arange(1, X.uns["Pf2_A"].shape[1] + 1)] 
    factors_only_df = cond_factors_df[factors_only]

 
    
    #Converting categorical feature into a numeric array
    y_category = cond_factors_df[feature]
    le = preprocessing.LabelEncoder()
    transformed_y = le.fit_transform(y_category)
    
    # print(transformed_y)
    # y_category_df = cond_factors_df[y_category]

    # y_category = y_category_df.to_numpy()
    #Converting boolean to int
    # y_category = np.multiply(y_category, 1) 

    
    #Series with labels for each patient characteristic 
    y_labels = pd.Series(transformed_y, index = cond_factors_df.index)

    predictions, coefficients = run_lr(factors_only_df, y_labels, proba = False)

    # AUC- ROC Curves
    fpr, tpr, _ = roc_curve(y_labels, predictions)
    roc_auc = roc_auc_score(y_labels, predictions)
    
    print(fpr)
    print(tpr)

    # # Separating into 0 and 1 categories
    # y_category_0 = y_category_df[y_category == 0]
    # y_category_1 = y_category_df[y_category == 1]  
    # print("Count of 0s:", len(y_category_0))
    # print("Count of 1s:", len(y_category_1))

    #Smoking status
    # y2_category = cond_factors_df[feature_2]
    # y2_category = y2_category.to_numpy()

    # y2_category_encoded = le.fit_transform(y2_category)
    # y2_category_encoded = pd.Series(y2_category_encoded)
    
    # print(y2_category_encoded)
    # print(dict(zip(le.classes_, le.transform(le.classes_))))

    #Run LR

    #Calculate accuracy
    accuracy = accuracy_score(y_labels, predictions)
    print(accuracy)
    


    # Plot ROC curve
    ax[0].plot(fpr, tpr, color='b', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    ax[0].plot([0,1], [0,1], color='k', lw=2,  linestyle='--', label=f'ROC Curve (AUC = 0.5)')
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_title(f'Receiver Operating Characteristic (ROC) Curve\nAccuracy: {np.round(accuracy, 3)}%')
    ax[0].legend(loc="lower right")
   
    
    return f 

SKF = StratifiedKFold(n_splits = 10)

#Running LR 
def run_lr(data: pd.DataFrame, labels: pd.Series, proba: bool = False
) -> tuple[pd.Series, pd.Series]:
    """
    Predicts categorical features via logistic regression cross-validation
    """
    #scale data
    data[:] = scale(data)
    lr_model = LogisticRegression()
    coefficients = pd.Series(0, index = data.columns, dtype = float)

    #Feature selector to find optimal number of features 
    rfe_cv = RFECV(lr_model, step = 1, cv= SKF)
    rfe_cv.fit(data, labels)
    data = data.loc[:, rfe_cv.support_]

    #Cross-validation
    probabilities = pd.Series(0, dtype = float, index = data.index)    
    for train_index, test_index in SKF.split(data, labels):
        train_group_data = data.iloc[train_index,:]
        train_labels = labels.iloc[train_index]
        test_group_data = data.iloc[test_index]

        lr_model.fit(train_group_data, train_labels)
        predicted = lr_model.predict_proba(test_group_data)
        probabilities[test_index] = predicted[:, 1]


     # AUC- ROC Curves
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = roc_auc_score(labels, probabilities)
    
    #Refitted on the entire dataset, retrieve coefficients 
    lr_model.fit(data, labels)
    coefficients.loc[rfe_cv.support_] = lr_model.coef_.squeeze()

    if proba:
        return probabilities, coefficients
        # pd.Series(probabilities, index = labels.index), pd.Series(coefficients, index = rfe_cv.support_)
    else:
        predicted = np.round(probabilities).astype(int)
        return pd.Series(predicted, index = labels.index), pd.Series(coefficients, index=np.argwhere(rfe_cv.support_).flatten() + 1)

    


# def predict_mortality(data: pd.DataFrame, meta: pd.DataFrame, proba: bool = False):
#     """
#     Predicts mortality status via cross-validation.
#     """

#     if not isinstance(data, pd.DataFrame):
#         data = pd.DataFrame(data)
    
#     data = data.loc[meta.loc[:, "patient_category"] != "Non-Pneumonia Control", :]
#     meta = meta.loc[meta.loc[:, "patient_category"] != "Non-Pneumonia Control", :]
#     labels = data.index.to_series().replace(meta.loc[:, "binary_outcome"])
#     groups = meta.loc[:, "patient_category"] == "COVID-19"
#     groups = groups.astype(int)


#     predictions = pd.Series(index = data.index)
#     coefficients = pd.DataFrame(index = data.columns, columns = np.unique(groups))

#     for group in groups.unique():
#         group_data = data.loc[groups == group, :]
#         group_labels = labels.loc[groups == group]

#         group_predictions, group_coefficients = run_lr(
#             group_data, group_labels, proba= proba
#         )
#         predictions.loc[groups == group] = group_predictions
#         coefficients.loc[:, group] = group_coefficients

#     coefficients.columns = ["Non-COVID", "COVID-19"]

#     if proba:
#         return predictions, labels
#     else:
#         accuracy = accuracy_score(labels, predictions)
#         return accuracy, coefficients


# #Plot Balanced Accuracy
# #Plot False Positive Rate vs. True Positive Rate

    
    


    
    

