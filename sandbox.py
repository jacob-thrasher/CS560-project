import numpy as np
from utils.survival_utils import concordance_impurity
from lifelines.utils import concordance_index
import random



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


times, events, predictions = perfect_case()
C = concordance_index(times, predictions, events)

groups = []
for i in range(20):
    r = random.random()
    if r < .5: groups.append('male')
    else: groups.append('female')

CI, counter = concordance_impurity(predictions, times, events, groups)
print(CI)

times, events, predictions = zero_case()
groups = []
for i in range(20):
    r = random.random()
    if r < .5: groups.append('male')
    else: groups.append('female')

CI, counter = concordance_impurity(predictions, times, events, groups)
print(CI)

# C = concordance_index(times, predictions, events)
# print(C)

times, events, predictions = half_case()
groups = []
for i in range(20):
    r = random.random()
    if r < .5: groups.append('male')
    else: groups.append('female')

CI, counter = concordance_impurity(predictions, times, events, groups)
print(CI)


# C = concordance_index(times, predictions, events)
# print(C)

