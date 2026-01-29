import os
import random
import pickle
import scipy
import torch
import numpy as np
import pandas as pd
import torchio as tio
import torchvision.transforms as T
import nibabel as nib
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 
from pycox.datasets import metabric, support, gbsg, flchain
from torch.utils.data import Dataset

def split_df(df):
    df_test = df.sample(frac=0.2)
    df.drop(df_test.index)
    df_valid = df.sample(frac=0.2)
    df.drop(df_valid.index)

    return df, df_valid, df_test

def centerCrop(img, length, width, height):
    assert img.shape[1] >= length
    assert img.shape[2] >= width
    # assert img.shape[3] >= height

    # T2 MRIs only have 42 slices
    if img.shape[3] < height:
        transform = tio.transforms.Pad((length, width, height))
        img = transform(img)

    x = img.shape[1]//2 - length//2
    y = img.shape[2]//2 - width//2
    z = img.shape[3]//2 - height//2
    img = img[:,x:x+length, y:y+width, z:z+height]
    return img

def randomCrop(img, length, width, height):
    assert img.shape[1] >= length
    assert img.shape[2] >= width
    # assert img.shape[3] >= height

    # T2 MRIs only have 42 slices
    if img.shape[3] < height:
        transform = tio.transforms.Pad((length, width, height))
        img = transform(img)


    x = random.randint(0, img.shape[1] - length)
    y = random.randint(0, img.shape[2] - width)
    z = random.randint(0, img.shape[3] - height )
    img = img[:,x:x+length, y:y+width, z:z+height]
    return img



def load_dataset(root, dataset, n_bins=10, time_mode='discrete', drop_cols=[], drop_sex=0, drop_race=0, drop_educ=0):
    '''
    Loads and converts desired dataset to a survival dataset

    Args:
        root - Path to cached dataset
        dataset - name of dataset to load
    
    Keyword args:
        dim    - Desired dimension
        n_bins - Desired number of bins to discretize continuous data (Default: 10)
    '''

    if dataset == 'NACC':
        train_surv = NACCDataset(os.path.join(root, 'train.csv'), time_mode=time_mode, name='NACC', drop_cols=drop_cols, drop_sex=drop_sex, drop_race=drop_race, drop_educ=drop_educ)
        valid_surv = NACCDataset(os.path.join(root, 'valid.csv'), time_mode=time_mode, name='NACC', drop_cols=drop_cols, drop_sex=drop_sex, drop_race=drop_race, drop_educ=drop_educ)
        test_surv = NACCDataset(os.path.join(root, 'test.csv')  , time_mode=time_mode, name='NACC', drop_cols=drop_cols, drop_sex=drop_sex, drop_race=drop_race, drop_educ=drop_educ)
        time_steps = np.arange(0, n_bins, 1)
 
    elif dataset in ['METABRIC', 'SUPPORT', 'GBSG', 'FLCHAIN']:
        if dataset == 'METABRIC': # 9
            df = metabric.read_df()
            cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']
            cols_leave = ['x4', 'x5', 'x6', 'x7']

        elif dataset == 'SUPPORT': # 14
            df = support.read_df()
            cols_standardize = ['x0', 'x2', 'x3', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']
            cols_leave = ['x1', 'x4', 'x5']

        elif dataset == 'GBSG':
            df = gbsg.read_df()
            cols_standardize = ['x1', 'x3', 'x4', 'x5', 'x6']
            cols_leave = ['x0', 'x2']

        elif dataset == 'FLCHAIN':
            # Note: Will raise error in pycox==0.2.3 --> I edited the source code to remove Unnamed:0 from drop list
            df = flchain.read_df()
            df.drop(columns=['rownames'], inplace=True)
            cols_standardize = ['age', 'kappa', 'lambda', 'flc.grp', 'creatinine']
            cols_leave = ['sex', 'mgus']
            df.rename(columns={'futime': 'duration', 'death': 'event'}, inplace=True)

        df_train, df_valid, df_test = split_df(df)

        # Standardize features
        standardize = [([col], StandardScaler()) for col in cols_standardize]
        leave = [(col, None) for col in cols_leave]
        x_mapper = DataFrameMapper(standardize + leave)
        x_train = x_mapper.fit_transform(df_train).astype('float32')
        x_valid = x_mapper.transform(df_valid).astype('float32')
        x_test = x_mapper.transform(df_test).astype('float32')

        labtrans = LabTransDiscreteTime(n_bins)
        get_target = lambda df: (df['duration'].values, df['event'].values)
        t_train, e_train = labtrans.fit_transform(*get_target(df_train))
        t_valid, e_valid = labtrans.transform(*get_target(df_valid))
        t_test, e_test = labtrans.transform(*get_target(df_test))


        train_surv = PyCoxDataset(x_train, t_train, e_train, name=dataset)
        valid_surv = PyCoxDataset(x_valid, t_valid, e_valid, name=dataset)
        test_surv = PyCoxDataset(x_test, t_test, e_test, name=dataset)
        time_steps = np.arange(0, n_bins, 1)

        
    return train_surv, valid_surv, test_surv, time_steps



class PyCoxDataset(Dataset):
    def __init__(self, features, times, events, name='Dataset'):

        self.features = features
        self.times = times
        self.events = events
        self.name = name

    
    def __len__(self):
        return len(self.times)
    
    def __getitem__(self, idx): 
        X = self.features[idx]
        t = self.times[idx]
        e = self.events[idx]

        return X, t, e, -1

class NACCDataset(Dataset):
    def __init__(self, df_path, time_mode='discrete', name="NACC", drop_cols=[], drop_sex=0, drop_race=0, drop_educ=0):

        assert time_mode in ['discrete', 'continuous'], f'Expected parameter time_mode to be one of [discrete, continuous], got {time_mode}'

        self.df = pd.read_csv(df_path)
        if len(drop_cols) > 0:
            self.df.drop(columns=drop_cols, inplace=True)

        self.time_mode = time_mode
        self.race_cols = [c for c in self.df.columns if "RACE" in c]
        self.name = name

        self.drop_sex, self.drop_race, self.drop_educ = drop_sex, drop_race, drop_educ

    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        cols_to_drop = ['NACCID', 'EVENT', 'TIME_TO_EVENT', 'TIME_TO_EVENT_DISCRETE']
        if self.drop_sex == 1: cols_to_drop.append('SEX')
        if self.drop_race == 1: cols_to_drop += self.race_cols
        if self.drop_educ == 1: cols_to_drop.append('EDUC')


        x = row.drop(cols_to_drop).tolist()
        e = row['EVENT']
        if self.time_mode == 'discrete':
            t = row['TIME_TO_EVENT_DISCRETE']
        elif self.time_mode == 'continuous':
            t = row['TIME_TO_EVENT']
        
        # Get subject profile
        race = row[self.race_cols]
        race = race[race == 1]
        if len(race) == 0: race = 'unknown'
        else: 
            race = race.index[0] 
            if   race == 'RACE_1.0': race = 'white'
            elif race == 'RACE_2.0': race = 'black'
            elif race == 'RACE_3.0': race = 'native_american'
            elif race == 'RACE_4.0': race = 'pacific_islander'
            elif race == 'RACE_5.0': race = 'asian'
            elif race == 'RACE_50.0': race = 'other'
            elif race == 'RACE_Missing': race = 'unknown'
            else: raise ValueError(f"Unknown race for subject {row['NACCID']}. Got {race}") 
            

        sex = 'male' if int(row['SEX']) == 0 else 'female'

        # Education: 
        educ = int(row['EDUC']*31)
        if educ <= 12: educ = 'high_school'
        elif educ <=16: educ = 'bachelors'
        elif educ <= 18: educ = 'masters'
        else: educ = 'doctoral'
        profile = {
            'naccid': row['NACCID'],
            'sex': sex,
            'race': race,
            'educ': educ
        }

        return torch.tensor(x, dtype=torch.float32), t, e, profile