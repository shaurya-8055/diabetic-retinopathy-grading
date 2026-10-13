# %% [markdown]
# # 03 · Ensemble + paired-eye fusion + screening operating point (CPU, no GPU quota)
#
# **Kaggle settings:** Accelerator **None**. *Add Input → Notebooks*: every `02_train_eval` run to combine.
# The notebook finds their `pred_val.csv` / `pred_test.csv` / `pred_external.csv` automatically.
# Optionally also attach `benjaminwarner/resized-2015-2019-blindness-detection-images`. Its `testLabels15.csv` marks the Kaggle **Private** test subset (42,670 images), the split the literature reports on.
#
# Every variant below is fitted on **val only**. Test and Messidor-2 stay untouched.
# * **Ensemble:** the mean of the models' expected-grade scores.
# * **Paired-eye fusion:** a linear model on [own score, other-eye score], refit on val. The two eyes of a patient have strongly correlated grades.
#   Huang et al. 2021 got a +2.7 % test-kappa gain from (feature-level) eye-pair fusion on EyePACS. Unpaired images (APTOS, Messidor-2) use their own score twice.
# * **Two operating points:**
#   1. QWK-optimal grade cut-offs, used for grading.
#   2. A **screening** cut-off on the score set so that referable DR (grade ≥ 2) has `SENS_TARGET` sensitivity on val, used for referral.
#      The QWK-optimal cut-offs trade sensitivity for kappa, which is the wrong trade for screening.

# %%
import json, os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, cohen_kappa_score, confusion_matrix, roc_auc_score

INPUT = Path(os.environ.get('DR_INPUT', '/kaggle/input'))
WORK = Path(os.environ.get('DR_WORK', '/kaggle/working'))
SENS_TARGET = 0.90  # referable-DR sensitivity the screening cut-off must reach on val
WORK.mkdir(parents=True, exist_ok=True)


def find(name, max_depth=4):
    return sorted(p for k in range(max_depth) for p in INPUT.glob('*/' * k + name))


RUNS = {p.parent.name: p.parent for p in find('pred_val.csv')}
TEST_LABELS = next(iter(find('testLabels15.csv', 6)), None)
print('runs:', list(RUNS), '\ntest labels:', TEST_LABELS)

# %%
KEY = ['shard', 'stem']


def load(split):
    base = None
    for name, d in RUNS.items():
        p = pd.read_csv(d / f'pred_{split}.csv')
        if base is None:
            base = p.drop(columns=['score', 'score_sd', 'pred'])
        base = base.merge(p[KEY + ['score']].rename(columns={'score': name}), on=KEY)
    return base


sets = {s: load(s) for s in ('val', 'test', 'external')}


def other_eye(d, col):  # score of the patient's other image; unpaired images fall back to their own score
    g = d.groupby('patient')[col]
    return np.where(g.transform('count') == 2, g.transform('sum') - d[col], d[col])


variants = {name: [name] for name in RUNS}
if len(RUNS) > 1:
    variants['ensemble'] = list(RUNS)
fusion = {}
for v, cols in list(variants.items()):
    for d in sets.values():
        d[v] = d[cols].mean(1)
    reg = LinearRegression().fit(np.c_[sets['val'][v], other_eye(sets['val'], v)], sets['val'].grade)
    fusion[v + '+pair'] = {'base': v, 'coef': reg.coef_.tolist(), 'intercept': float(reg.intercept_)}
    for d in sets.values():
        d[v + '+pair'] = reg.predict(np.c_[d[v], other_eye(d, v)])
    variants[v + '+pair'] = cols
print({k: np.round(f['coef'], 3).tolist() for k, f in fusion.items()}, '(weights on [own, other eye])')

if TEST_LABELS:  # fusion used both eyes of the full test set, so subset only after it
    u = pd.read_csv(TEST_LABELS)
    if 'Usage' in u.columns:
        t = sets['test'].merge(u[['image', 'Usage']].rename(columns={'image': 'stem'}), on='stem', how='left')
        sets['test_private'] = t[t.Usage == 'Private'].drop(columns='Usage').reset_index(drop=True)


