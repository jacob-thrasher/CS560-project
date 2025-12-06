import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import numba
from torch import nn
from scipy.interpolate import interp1d
from pycox.evaluation import EvalSurv

def cumsum_reverse(input: torch.Tensor, dim: int = 1) -> torch.Tensor:
    if dim != 1:
        raise NotImplementedError
    # input = input.sum(1, keepdim=True) - pad_col(input, where='start').cumsum(1)
    input = input.sum(1, keepdim=True) - input.cumsum(1)
    return input


def get_survival_curves(pred, method='DeepHit'):
    '''
    From: https://github.com/havakv/pycox/blob/master/pycox/models/pmf.py
    Get survival curve for batch of predictions
    '''

    if method == 'MTLR':
        pred = cumsum_reverse(pred, dim=1)    

    if method in ['DeepHit', 'PMF', 'MTLR', 'SurvRNC', 'RPS']:
        pmf = nn.functional.softmax(pred, dim=1)


    else: raise ValueError(f'Expected param method to be one of [PMF, MTLR, DeepHit, distance], got {method}')

    # Cumsum and inverse probs
    return 1 - torch.cumsum(pmf, dim=1)

@numba.jit(nopython=True)
def _is_comparable(t_i, t_j, e_i, e_j):
    return ((t_i < t_j) & e_i) | ((t_i == t_j) & (e_i | e_j))

def concordance_impurity(predictions, times, events, group, group_names=['male', 'female'], concordance='hazard'):

    assert concordance in ['hazard', 'td'], f'Expected parameter concordance to be one of [hazard, td], got {concordance}'

    counter = {}
    for g in group_names:
        counter[g] = {'num_comparable': 0, 'num_concordant': 0}

    for i in range(len(predictions)):
        for j in range(len(predictions)):
            if i == j: continue

            t_i, t_j = times[i], times[j]
            e_i, e_j = events[i], events[j]
            g_i, g_j = group[i], group[j]

            if concordance == 'hazard':
                r_i, r_j = predictions[i], predictions[j]
            elif concordance == 'td':
                # Invert survival probabilities to align with hazard
                r_i, r_j = 1 - predictions[i][t_i], 1 - predictions[j][t_i]

            # Ensure i and j are comparable
            if not _is_comparable(t_i, t_j, e_i, e_j): continue
            counter[g_i]['num_comparable'] += 1

            # Evaluate concordance
            if t_i < t_j:
                if r_i > r_j: counter[g_i]['num_concordant'] += 1
                elif r_i == r_j: counter[g_i]['num_concordant'] += 0.5
            elif t_i > t_j:
                if r_i < r_j: counter[g_i]['num_concordant'] += 1
                elif r_i == r_j: counter[g_i]['num_concordant'] += 0.5
            elif t_i == t_j:
                if e_i == 1 and e_j == 1:
                    if r_i == r_j: counter[g_i]['num_concordant'] += 1
                    else: counter[g_i]['num_concordant'] += 0.5
                elif e_i == 0 and e_j == 1 and r_i < r_j:
                    counter[g_i]['num_concordant'] += 1
                elif e_i == 1 and e_j == 0 and r_i > r_j:
                    counter[g_i]['num_concordant'] += 1
                else:
                    counter[g_i]['num_concordant'] += 0.5

    # Calculate concordance fraction (CF)
    for g in group_names:
        counter[g]['concordance_fraction'] = counter[g]['num_concordant'] / counter[g]['num_comparable']

    # Calculate CF devations between groups
    deviations = []
    for g_i in group_names:
        for g_j in group_names:
            if g_i == g_j: continue

            dev = abs(counter[g_i]['concordance_fraction'] - counter[g_j]['concordance_fraction'])
            deviations.append(dev)
    
    impurity_score = max(deviations)
    return impurity_score, counter


def get_metrics(predictions, time_range, times, events, method='DeepHit'):
    survival_curves = get_survival_curves(predictions, method=method)
    ev = EvalSurv(pd.DataFrame(survival_curves.T, time_range), np.array(times), np.array(events), censor_surv='km')
    return ev.concordance_td('antolini')