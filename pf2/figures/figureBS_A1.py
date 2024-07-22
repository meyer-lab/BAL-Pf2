import os
import anndata
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pacmap import PaCMAP
import pandas as pd
from pf2.data_import import condition_factors_meta
from pf2.tensor import correct_conditions
from pf2.figures.common import subplotLabel, getSetup

def makeFigure():
    # Directory to save the graphs and data
    output_dir = os.path.expanduser("~/BAL-Pf2/pf2/figures")
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

    # Initialize figure and axes setup
    ax, f = getSetup((8, 12), (2, 1))  # This must return both a figure and an axes array
    
    
    subplotLabel(ax)  

    # Load and preprocess data
    X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
    X.uns["Pf2_A"] = correct_conditions(X)
    cond_factors_df = condition_factors_meta(X)

    # Select component factors for PaCMAP
    factors_only = [f"Cmp. {i}" for i in np.arange(1, X.uns["Pf2_A"].shape[1] + 1)]
    factors_only_df = cond_factors_df[factors_only]

    # Perform PaCMAP dimensionality reduction
    pcm = PaCMAP(n_components=2)
    factors_only_umap = pcm.fit_transform(factors_only_df)

    # Create DataFrame for plotting
    umap_df = pd.DataFrame(data=factors_only_umap, columns=["PaCMAP1", "PaCMAP2"])
    umap_df["BAL_pct_macrophages"] = cond_factors_df["BAL_pct_macrophages"].to_numpy()

    # Plotting
    sns.scatterplot(data=umap_df, x="PaCMAP1", y="PaCMAP2", ax=ax[0])
    sns.scatterplot(data=umap_df, x="PaCMAP1", y="PaCMAP2", hue="BAL_pct_macrophages", ax=ax[1])

    # Save the figure
    plot_path = os.path.join(output_dir, "pacmap_BAL_pct_macrophages_plot.png")
    plt.savefig(plot_path)
    plt.close(f)

    # Save the DataFrame as a CSV file
    csv_path = os.path.join(output_dir, "pacmap_BAL_pct_macrophages_data.csv")
    umap_df.to_csv(csv_path, index=False)

    return f

# Execute the function
makeFigure()






#BELOW I USED THIS CODE TO CALCULATE THE NAN VALUES AND BOOLEAN VALUES AND SAVE AS A CSV

#import anndata
#import numpy as np
#import pandas as pd
#from pf2.data_import import condition_factors_meta
#from pf2.tensor import correct_conditions

# Load data
#X = anndata.read_h5ad("/opt/northwest_bal/full_fitted.h5ad")
#X.uns["Pf2_A"] = correct_conditions(X)
#cond_factors_df = condition_factors_meta(X)

# Show the boolean DataFrame for cond_factors_df
#boolean_df = cond_factors_df.isnull()
#print("\nShow the boolean DataFrame for cond_factors_df:\n\n", boolean_df)

# Save boolean DataFrame to CSV for easier inspection
#boolean_df.to_csv("boolean_dataframe.csv")

# Count total NaN in cond_factors_df
#total_nan_count = cond_factors_df.isnull().sum().sum()
#print("\nCount total NaN in cond_factors_df:\n\n", total_nan_count)

# Count NaN in each feature of cond_factors_df
#nan_count_per_feature = cond_factors_df.isnull().sum()
#print("\nCount NaN in each feature of cond_factors_df:\n\n", nan_count_per_feature)

# Save NaN counts per feature to CSV for easier inspection
#nan_count_per_feature.to_csv("nan_count_per_feature.csv", header=["NaN Count"])

# Count boolean values (True/False) for each feature
##boolean_counts = cond_factors_df.isnull().apply(pd.Series.value_counts).fillna(0).astype(int)
#print("\nBoolean counts (True/False) for each feature of cond_factors_df:\n\n", boolean_counts)

# Save boolean counts to CSV for easier inspection
#boolean_counts.to_csv("boolean_counts.csv")


# I WILL USE THE BELOW CODE TO USE AN IMPUTATION TEST FOR THE DATASET FOR FEATURES WITH LESS THAN 50 MISSING VALUES TO INFER THE MISSING VALUES

#import pandas as pd
#import numpy as np
#from sklearn.impute import SimpleImputer, KNNImputer
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error

# Load your data
#cond_factors_df = pd.read_csv('~/BAL-Pf2/cond_factors_data.csv')

# Identify feature columns dynamically (assuming features come after "Component_50")
#feature_columns = cond_factors_df.columns[50:]

# Separate numeric and non-numeric columns
#numeric_features = cond_factors_df[feature_columns].select_dtypes(include=[np.number]).columns
#non_numeric_features = cond_factors_df[feature_columns].select_dtypes(exclude=[np.number]).columns

# Convert boolean columns to object dtype to avoid issues with the imputer
#for feature in non_numeric_features:
    #if cond_factors_df[feature].dtype == 'bool':
        #cond_factors_df[feature] = cond_factors_df[feature].astype(object)

# Imputation Methods
#mean_imputer = SimpleImputer(strategy='mean')
#knn_imputer = KNNImputer(n_neighbors=5)
#most_frequent_imputer = SimpleImputer(strategy='most_frequent')

# Initialize a dictionary to store imputed dataframes
#imputed_dataframes = {
    #'mean': cond_factors_df.copy(),
    #'knn': cond_factors_df.copy(),
    #'most_frequent': cond_factors_df.copy()
#}

# Perform imputation for each feature column
#for feature in numeric_features:
    # Use mean and KNN imputation for numeric columns
    #imputed_dataframes['mean'][feature] = mean_imputer.fit_transform(cond_factors_df[[feature]]).flatten()
    #imputed_dataframes['knn'][feature] = knn_imputer.fit_transform(cond_factors_df[[feature]]).flatten()

#for feature in non_numeric_features:
    # Use most frequent imputation for non-numeric columns
    #imputed_dataframes['most_frequent'][feature] = most_frequent_imputer.fit_transform(cond_factors_df[[feature]]).flatten()

# Evaluation of imputation methods for numeric features
#mse_results = []

#for feature in numeric_features:
    # Define features and target for each feature separately
    #X = cond_factors_df.drop(columns=[feature])
    #y = cond_factors_df[feature]

    # Drop rows where the target feature is NaN for evaluation
    #not_null_indices = y.notnull()
    #X = X[not_null_indices]
    #y = y[not_null_indices]

    # Ensure that only numeric columns are used for the imputers
    #numeric_X = X.select_dtypes(include=[np.number])

    # Mean Imputation
    #X_mean_imputed = mean_imputer.fit_transform(numeric_X)
    #scores_mean = cross_val_score(LinearRegression(), X_mean_imputed, y, scoring='neg_mean_squared_error', cv=5)
    #mean_mse_mean_imputation = -scores_mean.mean()
    #mse_results.append({'Feature': feature, 'Imputation Method': 'Mean Imputation', 'Mean MSE': mean_mse_mean_imputation})

    # KNN Imputation
    #X_knn_imputed = knn_imputer.fit_transform(numeric_X)
    #scores_knn = cross_val_score(LinearRegression(), X_knn_imputed, y, scoring='neg_mean_squared_error', cv=5)
    #mean_mse_knn_imputation = -scores_knn.mean()
    #mse_results.append({'Feature': feature, 'Imputation Method': 'KNN Imputation', 'Mean MSE': mean_mse_knn_imputation})

# Convert mse_results to DataFrame and save to CSV
#mse_results_df = pd.DataFrame(mse_results)
#mse_results_df.to_csv("~/BAL-Pf2/imputation_mse_results_per_feature.csv", index=False)

# Save imputed datasets to CSV for easier inspection
#imputed_dataframes['mean'][numeric_features].to_csv("~/BAL-Pf2/mean_imputed_data.csv", index=False)
#imputed_dataframes['knn'][numeric_features].to_csv("~/BAL-Pf2/knn_imputed_data.csv", index=False)
#imputed_dataframes['most_frequent'][non_numeric_features].to_csv("~/BAL-Pf2/most_frequent_imputed_data.csv", index=False)




# THE BELOW CODE WITH GIVE ME THE R2
#import pandas as pd
#import numpy as np
#from sklearn.impute import SimpleImputer, KNNImputer
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error, r2_score

# Load your data
#input_file_path = '/home/ayanap/BAL-Pf2/cond_factors_data.csv'
#cond_factors_df = pd.read_csv("~/BAL-Pf2/cond_factors_data.csv")

# Identify feature columns dynamically (assuming features come after "Component_50")
#feature_columns = cond_factors_df.columns[50:]

# Separate numeric and non-numeric columns
#numeric_features = cond_factors_df[feature_columns].select_dtypes(include=[np.number]).columns
#non_numeric_features = cond_factors_df[feature_columns].select_dtypes(exclude=[np.number]).columns

# Convert boolean columns to object dtype to avoid issues with the imputer
#for feature in non_numeric_features:
    #if cond_factors_df[feature].dtype == 'bool':
        #cond_factors_df[feature] = cond_factors_df[feature].astype(object)

# Imputation Methods
#mean_imputer = SimpleImputer(strategy='mean')
#knn_imputer = KNNImputer(n_neighbors=5)
#most_frequent_imputer = SimpleImputer(strategy='most_frequent')

# Initialize a dictionary to store imputed dataframes
#imputed_dataframes = {
    #'mean': cond_factors_df.copy(),
    #'knn': cond_factors_df.copy(),
    #'most_frequent': cond_factors_df.copy()
#}

# Perform imputation for each feature column
#for feature in numeric_features:
    # Use mean and KNN imputation for numeric columns
    #imputed_dataframes['mean'][feature] = mean_imputer.fit_transform(cond_factors_df[[feature]]).flatten()
    #imputed_dataframes['knn'][feature] = knn_imputer.fit_transform(cond_factors_df[[feature]]).flatten()

#for feature in non_numeric_features:
    # Use most frequent imputation for non-numeric columns
    #imputed_dataframes['most_frequent'][feature] = most_frequent_imputer.fit_transform(cond_factors_df[[feature]]).flatten()

# Evaluation of imputation methods for numeric features
#mse_results = []
#r2_results = []

#for feature in numeric_features:
    # Define features and target for each feature separately
   # X = cond_factors_df.drop(columns=[feature])
    #y = cond_factors_df[feature]

    # Drop rows where the target feature is NaN for evaluation
    #not_null_indices = y.notnull()
   #X = X[not_null_indices]
    #y = y[not_null_indices]

    # Ensure that only numeric columns are used for the imputers
    #numeric_X = X.select_dtypes(include=[np.number])

    # Mean Imputation
    #X_mean_imputed = mean_imputer.fit_transform(numeric_X)
    #scores_mean = cross_val_score(LinearRegression(), X_mean_imputed, y, scoring='neg_mean_squared_error', cv=5)
   # mean_mse_mean_imputation = -scores_mean.mean()
    #mse_results.append({'Feature': feature, 'Imputation Method': 'Mean Imputation', 'Mean MSE': mean_mse_mean_imputation})

    # Calculate R² for mean imputation
    #model_mean = LinearRegression().fit(X_mean_imputed, y)
    #r2_mean = model_mean.score(X_mean_imputed, y)
    #r2_results.append({'Feature': feature, 'Imputation Method': 'Mean Imputation', 'R²': r2_mean})

    # KNN Imputation
    #X_knn_imputed = knn_imputer.fit_transform(numeric_X)
    #scores_knn = cross_val_score(LinearRegression(), X_knn_imputed, y, scoring='neg_mean_squared_error', cv=5)
    #mean_mse_knn_imputation = -scores_knn.mean()
    #mse_results.append({'Feature': feature, 'Imputation Method': 'KNN Imputation', 'Mean MSE': mean_mse_knn_imputation})

    # Calculate R² for KNN imputation
    #model_knn = LinearRegression().fit(X_knn_imputed, y)
    #r2_knn = model_knn.score(X_knn_imputed, y)
    #r2_results.append({'Feature': feature, 'Imputation Method': 'KNN Imputation', 'R²': r2_knn})

# Define output file paths
#output_dir = '/home/ayanap/BAL-Pf2/'
#mse_results_path = output_dir + "imputation_mse_results_per_feature.csv"
#r2_results_path = output_dir + "imputation_r2_results_per_feature.csv"
#mean_imputed_data_path = output_dir + "mean_imputed_data.csv"
#knn_imputed_data_path = output_dir + "knn_imputed_data.csv"
#most_frequent_imputed_data_path = output_dir + "most_frequent_imputed_data.csv"

# Convert mse_results to DataFrame and save to CSV
#mse_results_df = pd.DataFrame(mse_results)
#mse_results_df.to_csv(mse_results_path, index=False)

# Convert r2_results to DataFrame and save to CSV
#r2_results_df = pd.DataFrame(r2_results)
#r2_results_df.to_csv(r2_results_path, index=False)

# Save imputed datasets to CSV for easier inspection
#imputed_dataframes['mean'][numeric_features].to_csv(mean_imputed_data_path, index=False)
#imputed_dataframes['knn'][numeric_features].to_csv(knn_imputed_data_path, index=False)
#imputed_dataframes['most_frequent'][non_numeric_features].to_csv(most_frequent_imputed_data_path, index=False)



#THE CODE BELOW GIVES THE ABOVE SAME CSV BUT IT ALSO GRAPHS Y PREDICTED VS R TRUE

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.impute import SimpleImputer, KNNImputer
# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# import os

# # Load your data
# input_file_path = '/home/ayanap/BAL-Pf2/cond_factors_data.csv'
# cond_factors_df = pd.read_csv(input_file_path)

# # Identify feature columns dynamically (assuming features come after "Component_50")
# feature_columns = cond_factors_df.columns[50:]

# # Separate numeric and non-numeric columns
# numeric_features = cond_factors_df[feature_columns].select_dtypes(include=[np.number]).columns
# non_numeric_features = cond_factors_df[feature_columns].select_dtypes(exclude=[np.number]).columns

# # Convert boolean columns to object dtype to avoid issues with the imputer
# for feature in non_numeric_features:
#     if cond_factors_df[feature].dtype == 'bool':
#         cond_factors_df[feature] = cond_factors_df[feature].astype(object)

# # Imputation Methods
# mean_imputer = SimpleImputer(strategy='mean')
# knn_imputer = KNNImputer(n_neighbors=5)
# most_frequent_imputer = SimpleImputer(strategy='most_frequent')

# # Initialize a dictionary to store imputed dataframes
# imputed_dataframes = {
#     'mean': cond_factors_df.copy(),
#     'knn': cond_factors_df.copy(),
#     'most_frequent': cond_factors_df.copy()
# }

# # Perform imputation for each feature column
# for feature in numeric_features:
#     # Use mean and KNN imputation for numeric columns
#     imputed_dataframes['mean'][feature] = mean_imputer.fit_transform(cond_factors_df[[feature]]).flatten()
#     imputed_dataframes['knn'][feature] = knn_imputer.fit_transform(cond_factors_df[[feature]]).flatten()

# for feature in non_numeric_features:
#     # Use most frequent imputation for non-numeric columns
#     imputed_dataframes['most_frequent'][feature] = most_frequent_imputer.fit_transform(cond_factors_df[[feature]]).flatten()

# # Evaluation of imputation methods for numeric features
# mse_results = []
# r2_results = []

# # Prepare directory for plots
# plots_dir = '~/BAL-Pf2/R2Plots'
# os.makedirs(plots_dir, exist_ok=True)

# for feature in numeric_features:
#     # Define features and target for each feature separately
#     X = cond_factors_df.drop(columns=[feature])
#     y = cond_factors_df[feature]

#     # Drop rows where the target feature is NaN for evaluation
#     not_null_indices = y.notnull()
#     X = X[not_null_indices]
#     y = y[not_null_indices]

#     # Ensure that only numeric columns are used for the imputers
#     numeric_X = X.select_dtypes(include=[np.number])

#     # Mean Imputation
#     X_mean_imputed = mean_imputer.fit_transform(numeric_X)
#     scores_mean = cross_val_score(LinearRegression(), X_mean_imputed, y, scoring='neg_mean_squared_error', cv=5)
#     mean_mse_mean_imputation = -scores_mean.mean()
#     mse_results.append({'Feature': feature, 'Imputation Method': 'Mean Imputation', 'Mean MSE': mean_mse_mean_imputation})

#     # Calculate R² for mean imputation
#     model_mean = LinearRegression().fit(X_mean_imputed, y)
#     y_pred_mean = model_mean.predict(X_mean_imputed)
#     r2_mean = r2_score(y, y_pred_mean)
#     r2_results.append({'Feature': feature, 'Imputation Method': 'Mean Imputation', 'R²': r2_mean})

#     # KNN Imputation
#     X_knn_imputed = knn_imputer.fit_transform(numeric_X)
#     scores_knn = cross_val_score(LinearRegression(), X_knn_imputed, y, scoring='neg_mean_squared_error', cv=5)
#     mean_mse_knn_imputation = -scores_knn.mean()
#     mse_results.append({'Feature': feature, 'Imputation Method': 'KNN Imputation', 'Mean MSE': mean_mse_knn_imputation})

#     # Calculate R² for KNN imputation
#     model_knn = LinearRegression().fit(X_knn_imputed, y)
#     y_pred_knn = model_knn.predict(X_knn_imputed)
#     r2_knn = r2_score(y, y_pred_knn)
#     r2_results.append({'Feature': feature, 'Imputation Method': 'KNN Imputation', 'R²': r2_knn})

#     # Plot original vs imputed values for mean imputation
#     plt.figure(figsize=(8, 6))
#     plt.scatter(y, y_pred_mean, alpha=0.5)
#     plt.xlabel('True Values')
#     plt.ylabel('Predicted Values')
#     plt.title(f'{feature} - Mean Imputation')
#     plt.savefig(f'{plots_dir}{feature}_mean_imputation.png')
#     plt.show()

#     # Plot original vs imputed values for KNN imputation
#     plt.figure(figsize=(8, 6))
#     plt.scatter(y, y_pred_knn, alpha=0.5)
#     plt.xlabel('True Values')
#     plt.ylabel('Predicted Values')
#     plt.title(f'{feature} - KNN Imputation')
#     plt.savefig(f'{plots_dir}{feature}_knn_imputation.png')
#     plt.show()

# # Define output file paths
# output_dir = '~/BAL_Pf2/R2Plots/'
# mse_results_path = output_dir + "imputation_mse_results_per_feature.csv"
# r2_results_path = output_dir + "imputation_r2_results_per_feature.csv"
# mean_imputed_data_path = output_dir + "mean_imputed_data.csv"
# knn_imputed_data_path = output_dir + "knn_imputed_data.csv"
# most_frequent_imputed_data_path = output_dir + "most_frequent_imputed_data.csv"

# # Convert mse_results to DataFrame and save to CSV
# mse_results_df = pd.DataFrame(mse_results)
# mse_results_df.to_csv(mse_results_path, index=False)

# # Convert r2_results to DataFrame and save to CSV
# r2_results_df = pd.DataFrame(r2_results)
# r2_results_df.to_csv(r2_results_path, index=False)

# # Save imputed datasets to CSV for easier inspection
# imputed_dataframes['mean'][numeric_features].to_csv(mean_imputed_data_path, index=False)
# imputed_dataframes['knn'][numeric_features].to_csv(knn_imputed_data_path, index=False)
# imputed_dataframes['most_frequent'][non_numeric_features].to_csv(most_frequent_imputed_data_path, index=False)



#My attempt to calculate r^2


#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.impute import SimpleImputer, KNNImputer
#from sklearn.model_selection import cross_val_score
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error, r2_score
#import os

# #def makeFigure():
#     main()

# #def main():
#     # Load your data
#     input_file_path = '~/BAL-Pf2/cond_factors_data.csv'
#     try:
#         cond_factors_df = pd.read_csv(input_file_path)
#         print("Data loaded successfully.")
#         print(f"DataFrame shape: {cond_factors_df.shape}")
#     except FileNotFoundError:
#         print(f"File not found: {input_file_path}")
#         return
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return

#     # Identify feature columns dynamically (assuming features come after "Component_50")
#     feature_columns = cond_factors_df.columns[50:]

#     # Separate numeric and non-numeric columns
#     numeric_features = cond_factors_df[feature_columns].select_dtypes(include=[np.number]).columns
#     non_numeric_features = cond_factors_df[feature_columns].select_dtypes(exclude=[np.number]).columns

#     # Convert boolean columns to object dtype to avoid issues with the imputer
#     for feature in non_numeric_features:
#         if cond_factors_df[feature].dtype == 'bool':
#             cond_factors_df[feature] = cond_factors_df[feature].astype(object)

#     # Imputation Methods
#     mean_imputer = SimpleImputer(strategy='mean')
#     knn_imputer = KNNImputer(n_neighbors=5)
#     most_frequent_imputer = SimpleImputer(strategy='most_frequent')

#     # Initialize a dictionary to store imputed dataframes
#     imputed_dataframes = {
#         'mean': cond_factors_df.copy(),
#         'knn': cond_factors_df.copy(),
#         'most_frequent': cond_factors_df.copy()
#     }

#     # Perform imputation for each feature column
#     for feature in numeric_features:
#         # Use mean and KNN imputation for numeric columns
#         imputed_dataframes['mean'][feature] = mean_imputer.fit_transform(cond_factors_df[[feature]]).flatten()
#         imputed_dataframes['knn'][feature] = knn_imputer.fit_transform(cond_factors_df[[feature]]).flatten()

#     for feature in non_numeric_features:
#         # Use most frequent imputation for non-numeric columns
#         imputed_dataframes['most_frequent'][feature] = most_frequent_imputer.fit_transform(cond_factors_df[[feature]]).flatten()

#     # Evaluation of imputation methods for numeric features
#     mse_results = []
#     r2_results = []

#     # Prepare directory for plots
#     plots_dir = '~/BAL-Pf2/slopeR2'
#     os.makedirs(plots_dir, exist_ok=True)

#     for feature in numeric_features:
#         # Define features and target for each feature separately
#         X = cond_factors_df.drop(columns=[feature])
#         y = cond_factors_df[feature]

#         # Drop rows where the target feature is NaN for evaluation
#         not_null_indices = y.notnull()
#         X = X[not_null_indices]
#         y = y[not_null_indices]

#         # Ensure that only numeric columns are used for the imputers
#         numeric_X = X.select_dtypes(include=[np.number])

#         # Mean Imputation
#         X_mean_imputed = mean_imputer.fit_transform(numeric_X)
#         scores_mean = cross_val_score(LinearRegression(), X_mean_imputed, y, scoring='neg_mean_squared_error', cv=5)
#         mean_mse_mean_imputation = -scores_mean.mean()
#         mse_results.append({'Feature': feature, 'Imputation Method': 'Mean Imputation', 'Mean MSE': mean_mse_mean_imputation})

#         # Calculate R² and y-intercept for mean imputation
#         model_mean = LinearRegression().fit(X_mean_imputed, y)
#         y_pred_mean = model_mean.predict(X_mean_imputed)
#         r2_mean = r2_score(y, y_pred_mean)
#         r2_results.append({'Feature': feature, 'Imputation Method': 'Mean Imputation', 'R²': r2_mean})
#         slope_mean = model_mean.coef_[0]
#         intercept_mean = model_mean.intercept_

#         # KNN Imputation
#         X_knn_imputed = knn_imputer.fit_transform(numeric_X)
#         scores_knn = cross_val_score(LinearRegression(), X_knn_imputed, y, scoring='neg_mean_squared_error', cv=5)
#         mean_mse_knn_imputation = -scores_knn.mean()
#         mse_results.append({'Feature': feature, 'Imputation Method': 'KNN Imputation', 'Mean MSE': mean_mse_knn_imputation})

#         # Calculate R² and y-intercept for KNN imputation
#         model_knn = LinearRegression().fit(X_knn_imputed, y)
#         y_pred_knn = model_knn.predict(X_knn_imputed)
#         r2_knn = r2_score(y, y_pred_knn)
#         r2_results.append({'Feature': feature, 'Imputation Method': 'KNN Imputation', 'R²': r2_knn})
#         slope_knn = model_knn.coef_[0]
#         intercept_knn = model_knn.intercept_

#         # Plot original vs imputed values for mean imputation
#         plt.figure(figsize=(8, 6))
#         plt.scatter(y, y_pred_mean, alpha=0.5)
#         plt.xlabel('True Values')
#         plt.ylabel('Predicted Values')
#         plt.title(f'{feature} - Mean Imputation')
#         plt.plot(y, slope_mean * y + intercept_mean, color='red')  # Plot regression line
#         plt.text(0.05, 0.95, f'y = {slope_mean:.2f}x + {intercept_mean:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
#         plt.savefig(f'{plots_dir}{feature}_mean_imputation.svg')
#         plt.show()

#         # Plot original vs imputed values for KNN imputation
#         plt.figure(figsize=(8, 6))
#         plt.scatter(y, y_pred_knn, alpha=0.5)
#         plt.xlabel('True Values')
#         plt.ylabel('Predicted Values')
#         plt.title(f'{feature} - KNN Imputation')
#         plt.plot(y, slope_knn * y + intercept_knn, color='red')  # Plot regression line
#         plt.text(0.05, 0.95, f'y = {slope_knn:.2f}x + {intercept_knn:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
#         plt.savefig(f'{plots_dir}{feature}_knn_imputation.svg')
#         plt.show()

#     # Define output file paths
#     output_dir = '~/BAL-Pf2/slopeR2'
#     mse_results_path = output_dir + "imputation_mse_results_per_feature.csv"
#     r2_results_path = output_dir + "imputation_r2_results_per_feature.csv"
#     mean_imputed_data_path = output_dir + "mean_imputed_data.csv"
#     knn_imputed_data_path = output_dir + "knn_imputed_data.csv"
#     most_frequent_imputed_data_path = output_dir + "most_frequent_imputed_data.csv"

#     # Convert mse_results to DataFrame and save to CSV
#     mse_results_df = pd.DataFrame(mse_results)
#     mse_results_df.to_csv(mse_results_path, index=False)

#     # Convert r2_results to DataFrame and save to CSV
#     r2_results_df = pd.DataFrame(r2_results)
#     r2_results_df.to_csv(r2_results_path, index=False)

#     # Save imputed datasets to CSV for easier inspection
#     imputed_dataframes['mean'][numeric_features].to_csv(mean_imputed_data_path, index=False)
#     imputed_dataframes['knn'][numeric_features].to_csv(knn_imputed_data_path, index=False)
#     imputed_dataframes['most_frequent'][non_numeric_features].to_csv(most_frequent_imputed_data_path, index=False)

# if __name__ == "__main__":
#     main ()