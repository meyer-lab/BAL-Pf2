import pandas as pd
from scipy.stats import pearsonr



def correlation_df(df: pd.DataFrame, meta: bool) -> pd.DataFrame:
    """Returns a dataframe with the correlation of the meta data only"""
    if meta is True:
        columns = correlates
    else:
        columns = df.columns
        
    pearson_df = pd.DataFrame(
        columns=columns,
        index=columns,
        dtype=float
    )

    for row in columns:
        for column in columns:
            two_df = df[[row, column]].dropna()         
            if len(two_df.values)>2:
                result = pearsonr(
                    two_df.iloc[:, 0].values,
                    two_df.iloc[:, 1].values
                )
                pearson_df.loc[row, column] = result.pvalue
                
    return pearson_df
        
        
def correlation_meta_cc_df(cell_comp_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    """Returns a dataframe with the correlation of the meta data and cell composition"""
    pearson_df = pd.DataFrame(
        columns=correlates,
        index=cell_comp_df.columns,
        dtype=float
    )
    
    for row in cell_comp_df.columns:
        for column in correlates:
            one_meta_df = meta_df[column].dropna()
            common = one_meta_df.index.intersection(cell_comp_df.index)
            one_meta_df = one_meta_df.loc[common]
            cell_comp_partial_df = cell_comp_df.loc[common, :]
            if len(one_meta_df.values)>2:
                result = pearsonr(
                    one_meta_df.values,
                    cell_comp_partial_df.loc[:, row].values
                )
                pearson_df.loc[row, column] = result.pvalue

    return pearson_df


correlates = [
    "age", "bmi", "cumulative_icu_days", "admit_sofa_score", "admit_aps_score",
    "cumulative_intubation_days", "BAL_amylase", "BAL_pct_neutrophils",
    "BAL_pct_macrophages", "BAL_pct_monocytes", "BAL_pct_lymphocytes",
    "BAL_pct_eosinophils", "BAL_pct_other",
    "temperature", "heart_rate", "systolic_blood_pressure",
    "diastolic_blood_pressure", "mean_arterial_pressure",
    "norepinephrine_rate", "respiratory_rate", "oxygen_saturation",
    "rass_score", "peep", "fio2", "plateau_pressure", "lung_compliance",
    "minute_ventilation", "abg_ph", "abg_paco2", "pao2fio2_ratio", "wbc_count",
    "bicarbonate", "creatinine", "albumin", "bilirubin", "crp", "d_dimer",
    "ferritin", "ldh", "lactic_acid", "procalcitonin", "nat_score",
    "steroid_dose", "episode_duration", "icu_day", "number_of_icu_stays",
    "gcs_eye_opening", "gcs_motor_response"
]

