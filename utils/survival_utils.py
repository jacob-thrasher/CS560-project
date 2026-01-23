import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import numba
from tqdm import tqdm
from torch import nn
from scipy.interpolate import interp1d
from pycox.evaluation import EvalSurv
from sksurv.nonparametric import kaplan_meier_estimator
from lifelines import KaplanMeierFitter

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
    return ((t_i < t_j) & e_i) | ((t_i == t_j) & e_i & e_j == 0)

def _not_comparable(t_i, t_j, e_i, e_j):
    return (t_i < t_j & e_i == 0) | (t_j < t_i & e_j == 0) | (t_i == t_j & e_i == 0 & e_j == 0)

def concordance_impurity(predictions, times, events, group, concordance='hazard'):

    assert concordance in ['hazard', 'td'], f'Expected parameter concordance to be one of [hazard, td], got {concordance}'

    group_names = list(set(group))

    counter = {}
    for g in group_names:
        counter[g] = {'num_comparable': 0, 'num_concordant': 0}

    for i in tqdm(range(len(predictions)), desc='Computing impurity'):
        for j in range(len(predictions)):

            t_i, t_j = times[i], times[j]
            e_i, e_j = events[i], events[j]
            g_i, g_j = group[i], group[j]

            if i == j: continue
            if g_i == 'unknown' or g_j == 'unknown': continue

            if concordance == 'hazard':
                r_i, r_j = predictions[i], predictions[j]
            elif concordance == 'td':
                # Invert survival probabilities to align with hazard
                r_i, r_j = 1 - predictions[i][t_i], 1 - predictions[j][t_i]

            # Ensure i and j are comparable
            if not _is_comparable(t_i, t_j, e_i, e_j): continue
            # if _not_comparable(t_i, t_j, e_i, e_j): continue
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
        if g == 'unknown': continue
        counter[g]['concordance_fraction'] = counter[g]['num_concordant'] / counter[g]['num_comparable']

    # Calculate CF devations between groups
    deviations = []
    for g_i in group_names:
        for g_j in group_names:
            if g_i == g_j: continue
            if g_i == 'unknown' or g_j == 'unknown': continue

            dev = abs(counter[g_i]['concordance_fraction'] - counter[g_j]['concordance_fraction'])
            deviations.append(dev)
    
    impurity_score = max(deviations)
    return impurity_score, counter

def boostrapped_km_cal(predictions, times, events, n_bootstrap=1000, method='DeepHit'):
    rng = np.random.default_rng(67)
    n = len(predictions)
    bootstrap_values = []

    kmf = KaplanMeierFitter()
    kmf.fit(durations=times, event_observed=events, timeline=list(range(0, 10)))
    g_km = torch.tensor(kmf.survival_function_['KM_estimate'])

    for _ in tqdm(range(n_bootstrap), desc='Bootstrapping KM Cal', disable=True):
        sample_indices = rng.choice(n, size=n, replace=True)
        p = torch.tensor(predictions)[sample_indices]

        survival_curves = get_survival_curves(p, method=method)
        g_surv = survival_curves.mean(dim=0)
        g_km_cal = ((g_km - g_surv) ** 2).mean()
        bootstrap_values.append(g_km_cal)

    return torch.tensor(bootstrap_values)



def km_fair_calibration(predictions, times, events, group, method='DeepHit'):
    # Construct group dictionary
    # There's def a better way to do this
    group_names = list(set(group))
    if 'unknown' in group_names: group_names.remove('unknown')
    if 'other' in group_names: group_names.remove('other')
    
    group_dict = {g: {
        'predictions': [],
        'times': [],
        'events': [],
        'KM_cal': -1
    } for g in group_names}
    

    for p, t, e, g in zip(predictions, times, events, group):
        if g in ['unknown', 'other']: continue
        group_dict[g]['predictions'].append(p)
        group_dict[g]['times'].append(t)
        group_dict[g]['events'].append(e)


    # Group-wise KM fair calibration
    for g in sorted(group_names):

        # kmf = KaplanMeierFitter()
        # kmf.fit(durations=group_dict[g]['times'], event_observed=group_dict[g]['events'], timeline=list(range(0, 10)))
        # g_km = torch.tensor(kmf.survival_function_['KM_estimate'])

        # survival_curves = get_survival_curves(torch.tensor(group_dict[g]['predictions']), method=method)
        # g_surv = survival_curves.mean(dim=0)

        # # g_km_cal = ((g_km - g_surv) ** 2).mean()
        # g_km_cal = ((g_km - g_surv) ** 2)
        # group_dict[g]['KM_cal'] = g_km_cal
        g_bootstrap_cal = boostrapped_km_cal(group_dict[g]['predictions'], group_dict[g]['times'], group_dict[g]['events'])
        group_dict[g]['KM_cal'] = g_bootstrap_cal.tolist()

    

    return group_dict

def km_cal(predictions, times, events, method='DeepHit'):
    kmf = KaplanMeierFitter()
    kmf.fit(durations=times, event_observed=events, timeline=list(range(0, 10)))
    km = torch.tensor(kmf.survival_function_['KM_estimate'])

    survival_curves = get_survival_curves(predictions, method=method)
    avg_surv = survival_curves.mean(dim=0)

    return ((km - avg_surv) ** 2).mean().item()


def get_metrics(predictions, time_range, times, events, method='DeepHit', sens_attribute=None, groups=None):
    survival_curves = get_survival_curves(predictions, method=method)
    ev = EvalSurv(pd.DataFrame(survival_curves.T, time_range), np.array(times), np.array(events), censor_surv='km')
    cal = km_cal(predictions, times, events)
    fair_cal = km_fair_calibration(predictions.tolist(), times, events, groups, method=method)
    impurity = -1
    counts = {}
    ibs = float(ev.integrated_brier_score(time_range))

    if sens_attribute: 
        assert groups is not None, 'Parameter groups cannot be None when sens_attribute is defined!'
        impurity, counts = concordance_impurity(predictions, times, events, groups, concordance='td')
        # impurity, counts = -1, -1
    
    
    results = {
        'C': ev.concordance_td('antolini'),
        'IBS': ibs,
        'KM_Cal': cal,
        f'Impurity_{sens_attribute}': impurity,
        f'CI_counts_{sens_attribute}': counts,
        f'fair_cal_{sens_attribute}': fair_cal
    }
    
    return results