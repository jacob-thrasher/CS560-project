import pandas as pd
import math
import numpy as np
from tqdm import tqdm
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 

def days_between_age(age1, age2):
    delta = age2 - age1
    return int(365 * delta)

def impute_missing(df, feature_details, continuous_mode='mean', cat_mode='missing'):
    continuous_features = feature_details[~feature_details['is_discrete']]['feature']
    discrete_features = feature_details[feature_details['is_discrete']]['feature']


    # Impute continuous
    # for c in continuous_features:
    #     missing = df[c].isna().astype(int)
    #     if sum(missing) > 0:
    #         df[f'{c}_was_missing'] = missing

    if continuous_mode == 'mean':
        df[continuous_features] = df[continuous_features].fillna(df[continuous_features].mean())
    elif continuous_mode == 'median':
        df[continuous_features] = df[continuous_features].fillna(df[continuous_features].median())

    
    # Impute discrete
    if cat_mode == 'missing':
        df[discrete_features] = df[discrete_features].fillna("Missing")
    elif cat_mode == 'mode':
        df[discrete_features] = df[discrete_features].fillna(df[discrete_features].mode())

    return df

def process_NACC_data(src, dst, feature_details=None, discrete_time_bins=10, min_perc=0.5, cat_impute='missing'):
    df = pd.read_csv(src)


    # Filter features
    features = [
        'NACCID',
        'NACCAGE',
        'SEX',
        'RACE',
        'EDUC',
        'NACCFAM',
        'OTHMUT',
        'OTHMUTX',
        'NACCADMU',
        'NACCGDS',
        'CDRSUM',
        'APA',
        'AGIT',
        'HYPERTEN',
        'DIABETES',
        'STROKE',
        'CBTIA',
        'CVD',
        'CVDIF',
        'TOBAC100',
        'SMOKYRS',
        'TOBAC30',
        'PACKSPER',
        'NACCBMI',
        'NACCMMSE',
        'NACCMOCA',
        'NORMCOG',
        'DEMENTED',
        'COMMUN',
        'HOMEHOBB',
        'JUDGMENT',
        'ORIENT',
        'MEMORY',
        'PERSCARE',
        'WEIGHT',
        'BPSYS',
        'HRATE',
        'BPDIAS',
        'INDEPEND',
        'CDRLANG',
        'MEMPROB',
        'COMPORT',
        'ANYMEDS',
        'SHOPPING',
        'MEALPREP',
        'GAMES',
        'EVENT',
        'TIME_TO_EVENT'
    ]
    df = df[features]

    # Truncate where \delta=1 and t=0
    trunc = (df['EVENT'] == 1) & (df['TIME_TO_EVENT'] == 0)
    print(f'Truncating: Dropping {len(trunc)} rows')
    df = df[~trunc]

    # Replace placeholders for missing values with nans (to unify missingness)
    details_df = pd.read_csv(feature_details)

    # details = details_df[details_df['feature'].isin(['OTHMUT', 'OTHMUTX'])]
    # Replace placeholder values with nans for imputation
    for _, row in details_df.iterrows():
        feature = row['feature']
        missing_labels = [row[c] for c in ['missing_label_1', 'missing_label_2', 'missing_label_3'] if not math.isnan(row[c])]
        if len(missing_labels) > 0:
            df[feature] = df[feature].replace(missing_labels, np.nan)

    # Drop columns with more than min_perc missing features (Do this after imputation to avoid key errors)
    nans = df.isna().sum()
    min_allowable = int(len(df) * (1 - min_perc))
    cols_to_drop = nans[nans > min_allowable].keys()

    # Impute nans
    df = impute_missing(df, details_df, continuous_mode='mean', cat_mode=cat_impute)
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"Dropping features: {list(cols_to_drop)} - exceed nan threshold ({min_allowable})")



    # One-hot encode discrete features
    features_to_onehot = []
    for feat in details_df[details_df['is_discrete']]['feature'].tolist():
        if feat in cols_to_drop: continue
        if len(df[feat].unique()) > 2 and feat != 'NACCID':
            features_to_onehot.append(feat)
    # features_to_onehot = details_df[details_df['is_discrete'] & ~details_df['is_binary']]['feature'].tolist()
    # features_to_onehot.remove('NACCID')
    df = pd.get_dummies(df, columns=features_to_onehot)

    # Convert sex from  {1, 2} to {0, 1} values
    df['SEX'] = df['SEX'] - 1

    # Convert binarys to int
    for c in df.columns:
        if df[c].dtype == 'bool': df[c] = df[c].astype(int)

    
    # Standardize columns
    # cols_standardize = ['NACCAGE', 'EDUC', 'CDRSUM', 'SMOKYRS', 'NACCBMI', 'NACCMMSE', 'NACCMOCA']
    cols_standardize = details_df[~details_df['is_discrete']]['feature'].tolist()
    cols_standardize = [c for c in cols_standardize if c not in cols_to_drop]
    for c in cols_standardize:
        df[c] = (df[c]-df[c].min()) / (df[c].max()-df[c].min())

    # Discretize time values
    labtrans = LabTransDiscreteTime(discrete_time_bins)
    get_target = lambda df: (df['TIME_TO_EVENT'].values, df['EVENT'].values)
    t, e = labtrans.fit_transform(*get_target(df))
    df['TIME_TO_EVENT_DISCRETE'] = t


    df.to_csv(dst)

    return df


def process_HABS_data(src, dst, discrete_time_bins=10, min_perc=0.5, cat_impute='missing'):



    return


def generate_survival_labels(df, time_col, outcome_col, sub_id_col):
    new_cols = ['index'] + list(df.columns) + ['EVENT', 'TIME_TO_EVENT']
    df_proc = pd.DataFrame(columns=new_cols)

    subjects = df[sub_id_col].unique()

    for i, sub in enumerate(tqdm(subjects)):
        history = df[df[sub_id_col] == sub].reset_index().sort_values(time_col)
        event = 1 if 1 in history['IMH_Alzheimers'].tolist() else 0

        if event == 1:
            ad_visits = history[history['IMH_Alzheimers'] == 1].reset_index().sort_values(time_col)
            age_at_event = ad_visits.iloc[0].Age # Get Age at first visit of AD diagnosis
        elif event == 0:
            age_at_event = history.iloc[-1].Age  # Get Age at time of censorship (last visit)

        age_at_baseline = history.iloc[0].Age
        time = days_between_age(age_at_baseline, age_at_event)
        baseline_features = history.iloc[0].tolist()
        row = baseline_features + [event, time]
        df_proc.loc[i] = row

    return df_proc









# HABS
# Event indicator: IMH_Alzheimers
# Visit: Visit_ID
df = pd.read_csv('/home/WVU-AD/jdt0025/Documents/data/HABS_HD/HABS-HD_Clinical_Data/RP_HD_7_Clinical.csv')
print(df['CDX_Demn'].value_counts())


# df_proc = generate_survival_labels(df, 'Age', 'IMH_Alzheimers', 'Med_ID')
# df_proc.to_csv('/home/WVU-AD/jdt0025/Documents/data/HABS_HD/HABS-HD_Clinical_Data/HABS_survival.csv', index=False)


# df = pd.read_csv('/home/WVU-AD/jdt0025/Documents/data/HABS_HD/HABS-HD_Clinical_Data/HABS_survival.csv')
# num_events = sum(df['EVENT'].tolist())
# prop_censored = (len(df) - num_events) / len(df)
# print(prop_censored)
# obs = df[df['EVENT'] == 1]
# after_trunc = obs[obs['TIME_TO_EVENT'] > 0] 
# print(after_trunc.head())
# print(len(after_trunc))


# df = process_NACC_data('data/firstvisit_features_last_event.csv', 
#                   'data/NACC_proc_standardized.csv', 
#                   feature_details='data/feature_details.csv', 
#                   min_perc=.7,
#                   cat_impute='mode')

# print('Total features:', len(df.columns) - 4)

# df_train = df.sample(frac=0.7)
# df_test = df.drop(df_train.index)
# df_valid = df_test.sample(frac=0.5)
# df_test = df_test.drop(df_valid.index)

# print(len(df_train), len(df_valid), len(df_test))
# df_train.to_csv('data/train.csv', index=False)
# df_valid.to_csv('data/valid.csv', index=False)
# df_test.to_csv('data/test.csv', index=False)

# df = pd.read_csv('data/NACC_')
# n_censored = len(df) - sum(df['EVENT'])
# prop_censored = n_censored / len(df)
# print("Prop censored:", prop_censored)