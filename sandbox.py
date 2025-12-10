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

root = '/home/WVU-AD/jdt0025/Documents/exp/CS560'
exps = os.listdir(root)

models = ['NLL', 'DeepHit', 'MTLR', 'RPS', 'RPS_Ranking']
results = {}
for model in models:
    C, IBS, CI_sex, CI_race = [], [], [], []
    for seed in [67, 68, 69]:
        model_path = os.path.join(root, f'NACC_{model}_Adam0.0001_{str(seed)}')
        with open(os.path.join(model_path, 'results.yaml'), 'r') as f:
            metrics = yaml.safe_load(f)
        C.append(metrics['C'])
        IBS.append(metrics['IBS'])
        CI_sex.append(metrics['Impurity_sex'])
        CI_race.append(metrics['Impurity_race'])
    results[model] = {
        'C': {
            'avg': float(np.mean(C)),
            'std': float(np.std(C))
        },
        'IBS': {
            'avg': float(np.mean(IBS)),
            'std': float(np.std(IBS))
        },
        'CI_sex': {
            'avg': float(np.mean(CI_sex)),
            'std': float(np.std(CI_sex))
        },
        'CI_race': {
            'avg': float(np.mean(CI_race)),
            'std': float(np.std(CI_race))
        }
    }

f = open('avg_results.csv', 'w')
writer = csv.writer(f)
header = ['model', 'C', 'IBS', 'CI-td (sex)', 'CI-td (race)']
writer.writerow(header)

for model in models:
    row = [model, 
           f"{results[model]['C']['avg']:0.4f} ({results[model]['C']['std']:0.3f})",
           f"{results[model]['IBS']['avg']:0.4f} ({results[model]['IBS']['std']:0.3f})",
           f"{results[model]['CI_sex']['avg']:0.4f} ({results[model]['CI_sex']['std']:0.3f})",
           f"{results[model]['CI_race']['avg']:0.4f} ({results[model]['CI_race']['std']:0.3f})"]
    writer.writerow(row)

with open('avg_results.yaml', 'w') as f:
    yaml.dump(results, f)
