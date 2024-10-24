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
    ax, f = getSetup((50, 50), (4, 4))
    subplotLabel(ax)
    
    
    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")

    meta = import_meta()
    conversions = convert_to_patients(X)

    patient_factor = pd.DataFrame(
        X.uns["Pf2_A"],
        index=conversions,
        columns=np.arange(X.uns["Pf2_A"].shape[1]) + 1,
    )
    meta = meta.loc[patient_factor.index, :]
    
    type_of_meta = ["Float", "FloatInt"]
    type_of_data = ["Combine", "Meta", "Factors"]
    
    total_df = pd.DataFrame([])
    ax_numb = 0 
    for j in type_of_data:
        for i in type_of_meta:
            
            accuracy_plsr(patient_factor, meta, ax[ax_numb], ax[ax_numb+1], type_of_meta=i, type_of_data=j, threshold=1)
            ax[ax_numb].set(title="C19 "+ j+" "+i)
            ax[ax_numb+1].set(title="nC19 "+ j+" "+i)
            ax_numb +=2
            ax[ax_numb]
            
        #     break
        # break
            # accuracies_df = accuracies_df.melt(value_vars=["Overall", "C19", "nC19"],var_name="Status", value_name="Accuracy")
            # if j == "Factors":
            #     ax[ax_numb].set(title=j+" S:"+str(samples)+" Var:" + str(var))
            # else:
                # accuracies_df["DataType"] = j+i+" S:"+str(samples)+" Var:" + str(var)
                
            # total_df = pd.concat([total_df, accuracies_df])
            if j == "Factors":
                break
            
            
    
    # sns.barplot(total_df, x="DataType", y="Accuracy", hue="Status")

    return f

    


def accuracy_plsr(patient_factor, meta, ax1, ax2, type_of_meta="Float", type_of_data="Combine", threshold=.9):
    """Run PLSR depending on the type of data"""

    patient_factor = pd.concat([patient_factor, meta], axis=1)
    if type_of_meta == "Float":
        combined_factors = patient_factor.loc[:, patient_factor.dtypes == np.float64]
    elif type_of_meta == "FloatInt":
        combined_factors = pd.concat([patient_factor.loc[:, patient_factor.dtypes == np.float64], patient_factor.loc[:, patient_factor.dtypes == np.int64]], axis=1)
        
    combined_factors["patient_category"] = patient_factor["patient_category"] 
    combined_factors["binary_outcome"] = patient_factor["binary_outcome"] 
    combined_factors.columns = combined_factors.columns.astype(str)

    combined_factors.dropna(thresh=threshold*len(combined_factors), axis=1, inplace=True)

    combined_factors.dropna(inplace=True)
    
    meta = combined_factors[["patient_category", "binary_outcome"]]
    
    combined_factors = combined_factors.drop(columns=["patient_category", "binary_outcome"])
    
    if type_of_data == "Meta":
        combined_factors = combined_factors.iloc[:, 50:]
    if type_of_data == "Factors":
        combined_factors = combined_factors.iloc[:, :50]

        
    
    accuracies, (covid_plsr, nc_plsr) = predict_mortality(
        combined_factors,
        meta,
    )
    both_plsr = [covid_plsr, nc_plsr]
    ax = [ax1, ax2]
    
    
    for i in range(2):
        ax[i].scatter(
            both_plsr[i].y_loadings_[0, 0],
            both_plsr[i].y_loadings_[0, 1],
            c="tab:red"
        )
        ax[i].scatter(
                both_plsr[i].x_loadings_[:, 0],
                both_plsr[i].x_loadings_[:, 1],
                facecolors="white",
                edgecolors="k"
            )
        for index, component in enumerate(both_plsr[i].coef_.index):
                ax[i].text(
                    both_plsr[i].x_loadings_[index, 0],
                    both_plsr[i].x_loadings_[index, 1] - 0.001,
                    ha="center",
                    ma="center",
                    va="center",
                    s=component
            )
        ax[i].set(
            xlabel="PLSR1",
            ylabel="PLSR2",
        )
    

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

    plsr = PLSRegression(
        n_components=n_components,
        scale=True,
        max_iter=int(1E5))
    
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