import numpy as np
from utils.survival_utils import concordance_impurity
from lifelines.utils import concordance_index
import random
from data import NACCDataset, load_dataset
import pandas as pd
from network import Network, load_pretrained_model
import yaml
from torch.utils.data import DataLoader
from utils.survival_utils import get_survival_curves
import os
import yaml
import csv
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm

def zero_case(n=20):
    # Event times strictly increasing
    times = np.arange(1, n + 1)
    events = np.ones(n, dtype=int)

    # Predictions perfectly match (higher risk = earlier death)
    # So reverse-sort the times
    predictions = times[::-1].astype(float)

    return times.tolist(), events.tolist(), predictions.tolist()


# --------------------------
# 2. C ≈ 0.0  (Worst possible)
# --------------------------

def perfect_case(n=20):
    times = np.arange(1, n + 1)
    events = np.ones(n, dtype=int)

    # Predictions are exactly the opposite order needed
    predictions = times.astype(float)

    return times.tolist(), events.tolist(), predictions.tolist()


# --------------------------
# 3. C ≈ 0.5 (Balanced)
# --------------------------

def half_case(n=20, seed=0):
    rng = np.random.default_rng(seed)

    # Event times strictly increasing → full comparability
    times = np.arange(1, n + 1)
    events = np.ones(n, dtype=int)

    # Build predictions such that:
    # half the pairs are concordant and half discordant.
    # Approach: shuffle the predictions randomly.
    # For large n, random order produces expected C = 0.5.
    predictions = rng.random(n)

    return times.tolist(), events.tolist(), predictions.tolist()


# times, events, predictions = perfect_case()
# C = concordance_index(times, predictions, events)

# groups = []
# for i in range(20):
#     r = random.random()
#     if r < .5: groups.append('male')
#     else: groups.append('female')

# CI, counter = concordance_impurity(predictions, times, events, groups)
# print(CI)

# times, events, predictions = zero_case()
# groups = []
# for i in range(20):
#     r = random.random()
#     if r < .5: groups.append('male')
#     else: groups.append('female')

# CI, counter = concordance_impurity(predictions, times, events, groups)
# print(CI)

# # C = concordance_index(times, predictions, events)
# # print(C)

# times, events, predictions = half_case()
# groups = []
# for i in range(20):
#     r = random.random()
#     if r < .5: groups.append('male')
#     else: groups.append('female')

# CI, counter = concordance_impurity(predictions, times, events, groups)
# print(CI)


# C = concordance_index(times, predictions, events)
# print(C)

# with open('figures/METABRIC_DeepHit_Adam0.0001/config.yaml', 'r') as f:
#     cfg = yaml.safe_load(f)
# model = load_pretrained_model('figures/METABRIC_DeepHit_Adam0.0001/best_model_C.pt', out_dim=10, cfg=cfg, device='cuda')

# train_surv, _, _, _ = load_dataset('data', 'METABRIC')
# dataloader = DataLoader(train_surv, batch_size=32)

# model.eval()
# X, t, e, _ = next(iter(dataloader))
# out, _ = model(X)


# time_range = np.arange(0, 10, 1)
# survival_curves = get_survival_curves(out).detach().numpy()

# groups = []
# for i in range(32):
#     r = random.random()
#     if r < .5: groups.append('male')
#     else: groups.append('female')

# CI, counter = concordance_impurity(survival_curves, t.int().tolist(), e.int().tolist(), groups, concordance='td')
# print(counter)
# print(CI)


# df = pd.DataFrame(survival_curves.T, time_range)
# print(len(df.values))
# print(df.values[0])

# Surv --> 2D list, where each surv[i] corresponds to the survival probability of all elements at time i
# surv_idx











# root = '/home/WVU-AD/jdt0025/Documents/exp/CS560'
# exps = os.listdir(root)
# agg = {'asian': 0,
#        'black': 0,
#        'native_american': 0,
#        'other': 0,
#        'white': 0,
#        'pacific_islander': 0}
# for seed in [67, 68, 69]:
#     with open(os.path.join(root, f'NACC_RPS_Adam0.0001_{str(seed)}', 'results.yaml'), 'r') as f:
#         metrics = yaml.safe_load(f)

#     for r in metrics['CI_counts_race'].keys():
#         if r == 'unknown': continue
#         agg[r] += metrics['CI_counts_race'][r]['concordance_fraction']

# for k in agg.keys():
#     agg[k] /= 3



# race = list(agg.keys())
# cf = [agg[r] for r in race]

# cf, race = zip(*sorted(zip(cf, race), reverse=True))

# plt.figure(dpi=300) 
# plt.bar(race, cf)
# plt.xticks(rotation=45)
# plt.title('Concordance Fration (RPS)')
# plt.tight_layout()
# plt.show()







root = '/home/WVU-AD/jdt0025/Documents/exp/CHASE/main'
models = ['NLL', 'DeepHit', 'MTLR', 'RPS', 'RPS_Ranking']
results = {}
for model in tqdm(models):
    C, IBS, KM_cal = [], [], []
    CI_sex, CI_race, CI_educ = [], [], []
    is_fair_cal_sex = []
    is_fair_cal_race = {}
    is_fair_cal_educ = {}
    for seed in [67, 68, 69]:
        model_path = os.path.join(root, f'NACC_{model}_Adam0.0001_{str(seed)}')
        with open(os.path.join(model_path, 'results_fix.yaml'), 'r') as f:
            metrics = yaml.safe_load(f)
        C.append(metrics['C'])
        IBS.append(metrics['IBS'])
        KM_cal.append(metrics['KM_Cal'])
        CI_sex.append(metrics['Impurity_sex'])
        CI_race.append(metrics['Impurity_race'])
        CI_educ.append(metrics['Impurity_educ'])
        
        km_cal_sex = metrics['fair_cal_sex']
        diff_boot = np.array(km_cal_sex['male']['KM_cal']) - np.array(km_cal_sex['female']['KM_cal'])
        conf = np.percentile(diff_boot, [2.5, 97.5])
        if conf[0] <= 0 and conf[1] >= 0: is_fair_cal_sex.append(0)
        elif conf[0] <= 0 and conf[1] <= 0: is_fair_cal_sex.append(-1)
        else: is_fair_cal_sex.append(1)

        km_cal_race = metrics['fair_cal_race']
        pairs = list(combinations(km_cal_race.keys(), 2))
        for p1, p2 in pairs:
            key = f'{p1}-{p2}'
            if key not in is_fair_cal_race.keys(): is_fair_cal_race[key] = []

            diff_boot = np.array(km_cal_race[p1]['KM_cal']) - np.array(km_cal_race[p2]['KM_cal'])
            conf = np.percentile(diff_boot, [2.5, 97.5])
            if conf[0] <= 0 and conf[1] >= 0: is_fair_cal_race[key].append(0)
            elif conf[0] <= 0 and conf[1] <= 0: is_fair_cal_race[key].append(-1)
            else: is_fair_cal_race[key].append(1)

        km_cal_educ = metrics['fair_cal_educ']
        pairs = list(combinations(km_cal_educ.keys(), 2))
        for p1, p2 in pairs:
            key = f'{p1}-{p2}'
            if key not in is_fair_cal_educ.keys(): is_fair_cal_educ[key] = []

            diff_boot = np.array(km_cal_educ[p1]['KM_cal']) - np.array(km_cal_educ[p2]['KM_cal'])
            conf = np.percentile(diff_boot, [2.5, 97.5])
            if conf[0] <= 0 and conf[1] >= 0: is_fair_cal_educ[key].append(0)
            elif conf[0] <= 0 and conf[1] <= 0: is_fair_cal_educ[key].append(-1)
            else: is_fair_cal_educ[key].append(1)


    results[model] = {
        'C': {
            'avg': float(np.mean(C)) * 100,
            'std': float(np.std(C)) * 100
        },
        'IBS': {
            'avg': float(np.mean(IBS)) * 100,
            'std': float(np.std(IBS)) * 100
        },
        'KM_cal': {
            'avg': float(np.mean(KM_cal)),
            'std': float(np.std(KM_cal))
        },
        'CI_sex': {
            'avg': float(np.mean(CI_sex)) * 100,
            'std': float(np.std(CI_sex)) * 100
        },
        'CI_race': {
            'avg': float(np.mean(CI_race)) * 100,
            'std': float(np.std(CI_race)) * 100
        },
        'CI_educ': {
            'avg': float(np.mean(CI_educ)) * 100,
            'std': float(np.std(CI_educ)) * 100
        },
        'km_fair_sex': float(np.mean(is_fair_cal_sex)),
        'km_fair_race': {},
        'km_fair_educ': {}
    }
    for key, value in is_fair_cal_race.items():
        results[model]['km_fair_race'][key] = np.mean(value)
    for key, value in is_fair_cal_educ.items():
        results[model]['km_fair_educ'][key] = np.mean(value)

f = open('avg_results_fix.csv', 'w')
writer = csv.writer(f)
header = ['model', 'C', 'IBS', 'KM-cal', 'CI-td (sex)', 'CI-td (race)', 'CI-td (educ)', 'km-fair (sex)']
km_fair_race = [key for key, _ in is_fair_cal_race.items()]
header += km_fair_race
km_fair_educ = [key for key, _ in is_fair_cal_educ.items()]
header += km_fair_educ
writer.writerow(header)


for model in models:
    row = [model, 
           f"{results[model]['C']['avg']:.2f} ({results[model]['C']['std']:.2f})",
           f"{results[model]['IBS']['avg']:.2f} ({results[model]['IBS']['std']:.2f})",
           f"{results[model]['KM_cal']['avg']:.2f} ({results[model]['KM_cal']['std']:.2f})",
           f"{results[model]['CI_sex']['avg']:.2f} ({results[model]['CI_sex']['std']:.2f})",
           f"{results[model]['CI_race']['avg']:.2f} ({results[model]['CI_race']['std']:.2f})",
           f"{results[model]['CI_educ']['avg']:.2f} ({results[model]['CI_educ']['std']:.2f})",
           f"{results[model]['km_fair_sex']:.2f}"]
    race_info = [f"{results[model]['km_fair_race'][key]:.4f}" for key in km_fair_race]
    row += race_info
    educ_info = [f"{results[model]['km_fair_educ'][key]:.4f}" for key in km_fair_educ]
    row += educ_info


    writer.writerow(row)

with open('avg_results.yaml', 'w') as f:
    yaml.dump(results, f)




# with open('/home/WVU-AD/jdt0025/Documents/exp/CHASE/NACC_DeepHit_Adam0.0001_67/results.yaml', 'r') as f:
#     metrics = yaml.safe_load(f)

# print(type(metrics['fair_cal_educ']['bachelors']['KM_cal']))


