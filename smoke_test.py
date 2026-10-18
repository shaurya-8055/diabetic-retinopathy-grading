"""CPU smoke test: fake Kaggle inputs (split zips, csv.zip, duplicates, a blank image) -> 01 -> 02 (tiny ViT) -> 03.

    python smoke_test.py      # needs numpy, opencv, pillow, pandas, scipy, scikit-learn, matplotlib, torch, timm
"""
import io, os, subprocess, sys, tempfile, zipfile, json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
root = Path(tempfile.mkdtemp())
inp, work = root / 'input', root / 'working'


def fundus(grade, h=600, w=900):
    img = np.zeros((h, w, 3), np.uint8)
    cv2.circle(img, (w // 2, h // 2), h // 2 - 5, (int(rng.integers(120, 200)), 60, 30), -1)
    cv2.circle(img, (w // 2 + int(rng.integers(-h // 3, h // 3)), h // 2 + int(rng.integers(-h // 4, h // 4))), h // 10, (230, 200, 120), -1)
    for _ in range(6):  # "vessels", so distinct images get distinct perceptual hashes
        p = [(w // 2 + int(rng.integers(-h // 2.5, h // 2.5)), h // 2 + int(rng.integers(-h // 2.5, h // 2.5))) for _ in range(2)]
        cv2.line(img, p[0], p[1], (80, 20, 20), 8)
    for _ in range(grade * 15):  # "lesions"
        cv2.circle(img, (int(rng.integers(w // 2 - 200, w // 2 + 200)), int(rng.integers(100, h - 100))), 3, (90, 10, 10), -1)
    return img


def jpg(img, ext='.jpeg'):
    return cv2.imencode(ext, img[..., ::-1])[1].tobytes()


# EyePACS-like competition data: train.zip split into byte parts, labels in a .csv.zip
comp = inp / 'competitions/diabetic-retinopathy-detection'
comp.mkdir(parents=True)
for split, n in [('train', 40), ('test', 10)]:
    buf, labels = io.BytesIO(), []
    with zipfile.ZipFile(buf, 'w') as z:
        for pid in range(n):
            for eye in ('left', 'right'):
                g = int(rng.integers(0, 5))
                z.writestr(f'{split}/{pid + 1000 * (split == "test")}_{eye}.jpeg', jpg(fundus(g)))
                labels.append((f'{pid + 1000 * (split == "test")}_{eye}', g))
        z.writestr(f'{split}/blank_left.jpeg', jpg(np.zeros((300, 400, 3), np.uint8)))  # no fundus -> skipped
        labels.append(('blank_left', 0))
    data = buf.getvalue()
    for i, k in enumerate(range(0, len(data), len(data) // 3 + 1)):
        (comp / f'{split}.zip.{i + 1:03d}').write_bytes(data[k:k + len(data) // 3 + 1])
    lab = pd.DataFrame(labels, columns=['image', 'level'])
    if split == 'train':
        with zipfile.ZipFile(comp / 'trainLabels.csv.zip', 'w') as z:
            z.writestr('trainLabels.csv', lab.to_csv(index=False))
    else:
        bw = inp / 'datasets/benjaminwarner/resized-2015-2019-blindness-detection-images/labels'
        bw.mkdir(parents=True)
        lab.assign(Usage=np.where(np.arange(len(lab)) % 4, 'Private', 'Public')).to_csv(bw / 'testLabels15.csv', index=False)

# APTOS: pngs incl. a consistent duplicate pair and a conflicting duplicate pair
ap = inp / 'competitions/aptos2019-blindness-detection'
(ap / 'train_images').mkdir(parents=True)
rows = []
for i in range(30):
    g = i % 5
    rows.append((f'a{i:03d}', g))
    cv2.imwrite(str(ap / f'train_images/a{i:03d}.png'), fundus(g, 700, 1000)[..., ::-1])
for dup, src, g in [('dup_same', 'a001', 1), ('dup_conflict', 'a002', 4)]:
    (ap / f'train_images/{dup}.png').write_bytes((ap / f'train_images/{src}.png').read_bytes())
    rows.append((dup, g))
pd.DataFrame(rows, columns=['id_code', 'diagnosis']).to_csv(ap / 'train.csv', index=False)

# Messidor-2: images + Google adjudicated grades (one ungradable)
me = inp / 'datasets/xyaustin/messidor2/messidor-2/images'
me.mkdir(parents=True)
rows = []
for i in range(12):
    g = i % 5
    cv2.imwrite(str(me / f'2005_{i}_PP.png'), fundus(g, 960, 1440)[..., ::-1])
    rows.append((f'2005_{i}_PP.png', g if i else np.nan, 0, int(i != 0)))
(inp / 'datasets/google-brain/messidor2-dr-grades').mkdir(parents=True)
pd.DataFrame(rows, columns=['image_id', 'adjudicated_dr_grade', 'adjudicated_dme', 'adjudicated_gradable']) \
    .to_csv(inp / 'datasets/google-brain/messidor2-dr-grades/messidor_data.csv', index=False)

env = {**os.environ, 'DR_INPUT': str(inp), 'DR_WORK': str(work), 'DR_DATA': str(work), 'DR_SMOKE': '1', 'MPLBACKEND': 'Agg'}
here = Path(__file__).parent
for script in ('01_preprocess.py', '02_train_eval.py'):
    subprocess.run([sys.executable, str(here / script)], env=env, check=True, cwd=root)

df = pd.read_csv(work / 'labels.csv')
assert set(df.split) == {'train', 'val', 'test', 'external'}, set(df.split)
assert 'blank_left' not in set(df.stem), 'blank image should be skipped'
assert len(df[df.dataset == 'messidor2']) == 11, 'ungradable Messidor image should be dropped'
kept = set(df[df.keep].stem)
assert not {'a002', 'dup_conflict'} & kept, 'conflicting duplicates must be dropped'
assert len({'a001', 'dup_same'} & kept) == 1, 'consistent duplicates keep exactly one copy'
assert df.groupby('patient').split.nunique().max() == 1, 'a patient must not straddle splits'
m = json.load(open(work / 'metrics.json'))
assert {'val', 'test', 'external'} <= set(m), m.keys()
for f in ('model.pt', 'cm_val.png', 'cm_external.png', 'pred_test.csv'):
    assert (work / f).exists(), f
assert 'rdr_sens@screen' in m['test'], m['test'].keys()

# 03: combine the real run with a second, noisier fake run (exercises the ensemble path)
run2 = root / 'run2'
run2.mkdir()
for split in ('val', 'test', 'external'):
    p = pd.read_csv(work / f'pred_{split}.csv')
    p.assign(score=(p.score + rng.normal(0, 0.3, len(p))).clip(0, 4)).to_csv(run2 / f'pred_{split}.csv', index=False)
final = root / 'final'
subprocess.run([sys.executable, str(here / '03_fusion_ensemble.py')], env={**env, 'DR_INPUT': str(root), 'DR_WORK': str(final)},
               check=True, cwd=root)
t = pd.read_csv(final / 'results_table.csv')
assert {'working', 'run2', 'ensemble', 'ensemble+pair'} <= set(t.variant), set(t.variant)
assert {'val', 'test', 'test_private', 'external'} == set(t.set), set(t.set)
assert (t[t.set == 'val']['rdr_sens@screen'] >= 0.9 - 1e-9).all(), 'screening cut-off must hit the val sensitivity target'
assert json.load(open(final / 'final_config.json'))['variant'] in set(t.variant)
print('SMOKE TEST OK', root)
