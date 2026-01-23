import os
import torch
import argparse
import torchvision.transforms as T
import numpy as np
import random
import csv
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pycox.evaluation import EvalSurv
from torch import nn
from torch.utils.data import DataLoader
from scipy.stats import wilcoxon
from itertools import combinations

from utils.exp_helper import set_seed, load_config
from data import load_dataset
from train_test import test_step
from network import load_pretrained_model


if __name__ == '__main__':


    exp_root = '/home/WVU-AD/jdt0025/Documents/exp/CHASE'
    folders = sorted(os.listdir(exp_root))
    for i, exp in enumerate(folders):
        print(f'\n[{i} / {len(folders)}]: Evaluating model {exp}...\n')
        exp_path = os.path.join(exp_root, exp)
        # exp_path = 'figures/NACC_DeepHit_Adam0.0001'
        cfg = load_config(os.path.join(exp_path, 'config.yaml'))

        set_seed(cfg['seed'])

        batch_size = 128

        




        ########################
        # Load data
        root = cfg['exp_details']['data_root']
        train_surv, valid_surv, test_surv, time_steps = load_dataset(root, cfg['exp_details']['dataset'], drop_cols=['DEMENTED', 'NORMCOG'])


        train_dataloader  = DataLoader(train_surv, batch_size=batch_size, shuffle=True, drop_last=True)
        test_dataloader   = DataLoader(test_surv, batch_size=batch_size, shuffle=False)

        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = cfg['primary_device']
        print(f'\nUsing {device} device\n')



        method = cfg['train_params']['loss_fn']
        method = 'RPS' if method in ['RPS_Ranking', 'NLL'] else method

        model = load_pretrained_model(os.path.join(exp_path, 'best_model_C.pt'), 
                                                out_dim=len(time_steps),
                                                cfg=cfg)
        model.eval()
        model.to(device)

        _, results = test_step(model, test_dataloader, loss_fn=None, device=device, time_step=time_steps, sens_attribute='sex', method=method)
        _, results2 = test_step(model, test_dataloader, loss_fn=None, device=device, time_step=time_steps, sens_attribute='race', method=method)
        _, results3 = test_step(model, test_dataloader, loss_fn=None, device=device, time_step=time_steps, sens_attribute='educ', method=method)

        # metrics = test_step(model, test_dataloader, loss_fn=None, device=device, time_step=time_steps, sens_attribute='race', method=method)

        
        # metrics = results2['fair_cal_educ']
        # pairs = list(combinations(metrics.keys(), 2))
        # for pair in pairs:
            
        #     diff_boot = metrics[pair[0]]['KM_cal'] - metrics[pair[1]]['KM_cal']
        #     ci_diff = np.percentile(diff_boot, [2.5, 97.5])

        #     print(pair, ci_diff)

        # print(results2['KM_Cal'])

        # raise ValueError

        results['Impurity_race']  = results2['Impurity_race']
        results['CI_counts_race'] = results2['CI_counts_race']
        results['fair_cal_race']  = results2['fair_cal_race']
        results['Impurity_educ']  = results3['Impurity_educ']
        results['CI_counts_educ'] = results3['CI_counts_educ']
        results['fair_cal_educ']  = results3['fair_cal_educ']
        with open(os.path.join(exp_path, 'results_educ.yaml'), 'w') as f:
            yaml.dump(results, f)

        # print('C  :', results['C'])
        # print('IBS:', results['IBS'])
        # print('\nCI (sex):', results['Impurity_sex'])
        # print('Details:\n', results['CI_counts_sex'])
        # print('\nCI (race):', results2['Impurity_race'])
        # print('Details:\n', results2['CI_counts_race'])

        


