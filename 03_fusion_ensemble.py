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


# %%
# Same definitions as 02_train_eval
def qwk(y, p):
    return cohen_kappa_score(y, p, weights='quadratic')


def to_grade(s, thr=(0.5, 1.5, 2.5, 3.5)):
    return np.digitize(s, thr)


def fit_thresholds(y, s):
    thr = [0.5, 1.5, 2.5, 3.5]
    for _ in range(3):
        for k in range(4):
            lo, hi = (thr[k - 1] if k else 0.0), (thr[k + 1] if k < 3 else 4.0)
            thr[k] = max(np.linspace(lo, hi, 41)[1:-1], key=lambda t: qwk(y, to_grade(s, thr[:k] + [t] + thr[k + 1:])))
    return thr


def metrics(y, s, thr, rdr_t):
    p = to_grade(s, thr)
    r, ref = y >= 2, s >= rdr_t
    recall = [(p[y == g] == g).mean() for g in range(5) if (y == g).any()]
    return {'qwk': qwk(y, p), 'acc': (y == p).mean(), 'macro_recall': np.mean(recall),
            'rdr_auc': roc_auc_score(r, s) if 0 < r.sum() < len(r) else np.nan,
            'rdr_sens@qwk': (p[r] >= 2).mean(), 'rdr_spec@qwk': (p[~r] < 2).mean(),
            'rdr_sens@screen': ref[r].mean(), 'rdr_spec@screen': (~ref[~r]).mean(),
            'vtdr_auc': roc_auc_score(y >= 3, s) if 0 < (y >= 3).sum() < len(y) else np.nan}


cut = {}
rows = []
for v in variants:
    y, s = sets['val'].grade.values, sets['val'][v].values
    cut[v] = (fit_thresholds(y, s), float(np.quantile(s[y >= 2], 1 - SENS_TARGET, method='lower')))  # 'lower': sens >= target
    for name, d in sets.items():
        rows.append({'variant': v, 'set': name, 'n': len(d), **metrics(d.grade.values, d[v].values, *cut[v])})
table = pd.DataFrame(rows)
table.to_csv(WORK / 'results_table.csv', index=False)
pd.set_option('display.width', 200)
print(table.pivot(index='variant', columns='set', values='qwk').round(4))
print(table.round(3).to_string(index=False))

# %% [markdown]
# ## Final variant: chosen by val QWK, full report with 95 % bootstrap CIs

# %%
best = max(variants, key=lambda v: table[(table.variant == v) & (table.set == 'val')].qwk.item())
thr, rdr_t = cut[best]
print('final variant:', best, 'models:', variants[best], 'grade cut-offs', np.round(thr, 3), 'screening cut-off', round(rdr_t, 3))

for name, d in sets.items():
    if name == 'val':
        continue
    y, s = d.grade.values, d[best].values
    rng = np.random.default_rng(0)
    boot = pd.DataFrame([metrics(y[i], s[i], thr, rdr_t) for i in rng.integers(0, len(y), (300, len(y)))])
    m = metrics(y, s, thr, rdr_t)
    print(f'\n===== {name} (n={len(y)}) =====')
    for k, val in m.items():
        lo, hi = boot[k].quantile([0.025, 0.975])
        print(f'{k:18s} {val:.3f}  [{lo:.3f}, {hi:.3f}]')
    p = to_grade(s, thr)
    print(classification_report(y, p, labels=range(5), digits=3, zero_division=0,
                                target_names=['0 none', '1 mild', '2 moderate', '3 severe', '4 PDR']))
    cm = confusion_matrix(y, p, labels=range(5))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay(cm).plot(ax=ax[0], colorbar=False, values_format='d')
    ConfusionMatrixDisplay(cm / cm.sum(1, keepdims=True).clip(1)).plot(ax=ax[1], colorbar=False, values_format='.2f')
    ax[0].set_title(f'{best} · {name}: counts')
    ax[1].set_title('row-normalised (per-grade recall)')
    fig.savefig(WORK / f'cm_final_{name}.png', dpi=120, bbox_inches='tight')
    plt.show()

json.dump({'variant': best, 'models': variants[best], 'fusion': fusion.get(best), 'grade_thresholds': thr,
           'screening_threshold': rdr_t, 'sens_target': SENS_TARGET}, open(WORK / 'final_config.json', 'w'), indent=1)
