# %% [markdown]
# # 02 · Ordinal DR grader: train, TTA evaluation, external validation, screening tool
#
# **Kaggle settings:** Accelerator **GPU T4 x2**, Internet **On** (to download timm weights).
# Input: the output of `01_preprocess`. Attach it via *Add Input → Notebooks*, or as a dataset made from that output.
#
# | stage | data |
# |---|---|
# | train | EyePACS-2015 train + APTOS-2019 (deduplicated), minus val |
# | val | 10 % patient-grouped holdout; used for checkpoint selection and threshold tuning |
# | internal test | EyePACS-2015 test (~53k, labels released after the competition) |
# | **external** | Messidor-2 with adjudicated grades (different country, camera and grading protocol); never used for tuning |
#
# **Method**
# * Backbone: ViT-B/16 at **512 px** (1024 tokens) by default, with pos-embeddings interpolated from 384. Any timm model works, e.g. an EfficientNet as the CNN baseline.
# * **Ordinal head:** 4 logits, one per cumulative probability P(grade > k), trained with BCE (Frank & Hall / CORAL-style).
#   The continuous score Σσ(logitₖ) is the expected grade. Grade cut-offs are tuned on val to maximise QWK.
# * **Imbalance:** sampling weight ∝ class_count^-α, with α = 0.5 (square-root rebalancing). Fully balanced sampling over-predicts the rare grades.
#   The threshold tuning on val, which keeps the natural class mix, then recalibrates the decision point.
# * **Label noise:** ordinal label smoothing, plus the APTOS duplicate-conflict removal from notebook 01. Evaluation uses the *adjudicated* Messidor-2 labels.
# * **TTA:** identity + h-flip + v-flip + 180° rotation. The TTA spread is also used as an uncertainty signal that sends an image to a human grader.

# %%
import functools, glob, json, os, time, zipfile
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, cohen_kappa_score, confusion_matrix, roc_auc_score

SMOKE = bool(os.environ.get('DR_SMOKE'))  # tiny CPU run that only exercises the code path
DATA = Path(os.environ.get('DR_DATA') or Path(sorted(glob.glob('/kaggle/input/**/labels.csv', recursive=True))[0]).parent)
WORK = Path(os.environ.get('DR_WORK', '/kaggle/working'))

MODEL = 'vit_base_patch16_384.augreg_in21k_ft_in1k'  # CNN baseline: 'tf_efficientnet_b4.ns_jft_in1k' with LR = 2e-4
IMG = 512             # input resolution. Microaneurysms are ~2-4 px here; avoid going below 448.
PREPROC = 'ben'       # 'ben' (Graham 2015) | 'clahe' | 'none'
EPOCHS, BS, LR = 10, 16, 5e-5  # LR applies to the backbone; the head gets 10x
SAMPLE_ALPHA = 0.5    # sampling weight ∝ count^-alpha: 0 = natural distribution, 1 = fully balanced
SMOOTH = 0.05         # ordinal label smoothing (robustness to label noise)
N_TTA = 4
GRAD_CKPT = False     # True if CUDA OOM (slower, much less memory)
TIME_BUDGET_H = 9.0   # stop training early so evaluation still fits in Kaggle's 12 h session
CKPT = None           # path to an existing model.pt to skip training (re-run eval/screening without spending GPU hours)
WORKERS = os.cpu_count()
if SMOKE:
    MODEL, IMG, EPOCHS, BS, WORKERS = 'vit_tiny_patch16_224', 64, 1, 4, 2

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
AMP = DEV == 'cuda'
WORK.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
print(DATA, DEV, torch.cuda.device_count(), 'GPU(s)')

# %%
df = pd.read_csv(DATA / 'labels.csv')
df = df[df.keep & df.grade.notna()].astype({'grade': int})
splits = {s: d.reset_index(drop=True) for s, d in df.groupby('split')}
print(pd.crosstab(df.split, df.grade, margins=True))


# %%
@functools.cache
def _shard(name, pid):  # one handle per worker process
    p = DATA / f'{name}.zip'
    return zipfile.ZipFile(p) if p.exists() else DATA / name  # Kaggle may auto-extract zips into datasets


def load(shard, stem):
    s = _shard(shard, os.getpid())
    data = s.read(f'{stem}.jpg') if isinstance(s, zipfile.ZipFile) else (s / f'{stem}.jpg').read_bytes()
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def enhance(img, how=PREPROC):
    s = img.shape[0]
    if how == 'ben':  # subtract local average colour: normalises illumination and camera differences, keeps lesions
        img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), s / 30), -4, 128)
    elif how == 'clahe':
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[..., 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    mask = np.zeros((s, s, 1), np.uint8)
    cv2.circle(mask, (s // 2, s // 2), int(s * 0.45), 1, -1)  # clip the rim, where the blur leaves a bright ring
    return np.where(mask, img, 128 if how == 'ben' else 0).astype(np.uint8)


def augment(img, rng):
    s = img.shape[0]
    if rng.random() < 0.5:
        img = img[:, ::-1]
    m = cv2.getRotationMatrix2D((s / 2, s / 2), rng.uniform(0, 360), rng.uniform(0.95, 1.1))
    img = cv2.warpAffine(np.ascontiguousarray(img), m, (s, s))
    return np.clip(img * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20), 0, 255).astype(np.uint8)  # exposure jitter


class DS(torch.utils.data.Dataset):
    def __init__(self, d, train):
        self.d, self.train = d, train

    def __len__(self):
        return len(self.d)

    def __getitem__(self, i):
        r = self.d.iloc[i]
        img = cv2.resize(load(r.shard, r.stem), (IMG, IMG), interpolation=cv2.INTER_AREA)
        if self.train:
            img = augment(img, np.random.default_rng())
        return torch.from_numpy(enhance(img)).permute(2, 0, 1), torch.tensor(r.grade, dtype=torch.float32)


def loader(d, train):
    sampler = None
    if train:
        w = d.grade.map(d.grade.value_counts() ** -SAMPLE_ALPHA).to_numpy(copy=True)
        sampler = torch.utils.data.WeightedRandomSampler(w, len(d))
    return torch.utils.data.DataLoader(DS(d, train), BS if train else 2 * BS, sampler=sampler, num_workers=WORKERS,
                                       pin_memory=AMP, drop_last=train, persistent_workers=WORKERS > 0)


# %%
def make_model():
    kw = dict(pretrained=not SMOKE, num_classes=4)
    try:
        m = timm.create_model(MODEL, img_size=IMG, **kw)  # ViT/Swin: resample position embeddings to IMG
    except TypeError:
        m = timm.create_model(MODEL, **kw)  # CNNs accept any size
    m.set_grad_checkpointing(GRAD_CKPT)
    return m.to(DEV)


model = make_model()
MEAN = torch.tensor(model.pretrained_cfg['mean'], device=DEV).view(1, 3, 1, 1) * 255
STD = torch.tensor(model.pretrained_cfg['std'], device=DEV).view(1, 3, 1, 1) * 255
net = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model


def norm(x):
    return (x.to(DEV, non_blocking=True).float() - MEAN) / STD


def ordinal_targets(y):  # grade -> [y>0, y>1, y>2, y>3], smoothed towards 0.5
    return (y[:, None] > torch.arange(4, device=y.device)).float() * (1 - SMOOTH) + SMOOTH / 2


def expected_grade(logits):  # E[grade] = sum_k P(grade > k), continuous in [0, 4]
    return torch.sigmoid(logits.float()).sum(1)


def qwk(y, p):
    return cohen_kappa_score(y, p, weights='quadratic')


def to_grade(s, thr=(0.5, 1.5, 2.5, 3.5)):
    return np.digitize(s, thr)


@torch.no_grad()
def predict(d, tta=N_TTA):
    """Returns (mean, std) of the expected grade over TTA views."""
    net.eval()
    out = []
    for x, _ in loader(d, train=False):
        x = norm(x)
        views = [x, x.flip(3), x.flip(2), x.flip((2, 3))][:tta]
        with torch.autocast(DEV, dtype=torch.float16, enabled=AMP):
            out.append(torch.stack([expected_grade(net(v)) for v in views], 1).cpu())
    s = torch.cat(out).numpy()
    return s.mean(1), s.std(1)


