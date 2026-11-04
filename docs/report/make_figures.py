"""Regenerate the report figures from the executed Kaggle notebooks and results/*.csv.

    python docs/report/make_figures.py      # run from the repository root
"""
import base64, io, json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

OUT = Path('docs/report/figures')
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e',
                     'ytick.color': '#52514e', 'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.5,
                     'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.dpi': 300})
# Categorical slots 1-3 of the validated reference palette (all-pairs CVD-safe); identity also carried by legend + labels
VIT, CNN, ENS = '#2a78d6', '#eb6834', '#1baf7a'
INK2 = '#52514e'
COL, FULL = 3.45, 7.1  # IEEE column / page width (inches)


def notebook_images(path):
    nb = json.load(open(path))
    return [np.asarray(Image.open(io.BytesIO(base64.b64decode(o['data']['image/png']))).convert('RGB'))
            for c in nb['cells'] for o in c.get('outputs', []) if 'image/png' in o.get('data', {})]


def ben(img):
    s = img.shape[0]
    out = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), s / 30), -4, 128)
    mask = np.zeros((s, s, 1), np.uint8)
    cv2.circle(mask, (s // 2, s // 2), int(s * 0.45), 1, -1)
    return np.where(mask, out, 128).astype(np.uint8)


# 1. Stored 768-px crops (one per grade, rendered by 01_preprocess) and their Ben Graham enhancement
pre = notebook_images('Executed-Notebooks/diabetic-retinopathy-preprocessing.ipynb')
strip = pre[0]
body = strip[50:]
cols = (body < 245).any(2).any(0)
edges = np.flatnonzero(np.diff(np.r_[0, cols.astype(int), 0]))
panels = [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a > 100]
rows = (body < 245).any(2).mean(1) > 0.5  # image rows (titles cover only a few columns)
r0, r1 = np.flatnonzero(rows)[[0, -1]]
crops = [cv2.resize(np.ascontiguousarray(body[r0:r1, a:b]), (256, 256), interpolation=cv2.INTER_AREA) for a, b in panels]
order = [1, 4, 3, 2, 0]  # panels show grades 4,0,3,2,1 -> reorder to 0..4
names = ['0 No DR', '1 Mild', '2 Moderate', '3 Severe', '4 PDR']
fig, ax = plt.subplots(2, 5, figsize=(FULL, 3.0))
for j, k in enumerate(order):
    ax[0, j].imshow(crops[k])
    ax[1, j].imshow(ben(crops[k]))
    ax[0, j].set_title(names[j], fontsize=8)
for a in ax.ravel():
    a.axis('off')
ax[0, 0].text(-0.08, 0.5, 'cropped', transform=ax[0, 0].transAxes, rotation=90, va='center', ha='right')
ax[1, 0].text(-0.08, 0.5, 'Ben Graham', transform=ax[1, 0].transAxes, rotation=90, va='center', ha='right')
fig.subplots_adjust(wspace=0.04, hspace=0.04)
fig.savefig(OUT / 'fig_samples.png')
Image.fromarray(pre[1][40:]).save(OUT / 'fig_dups.png')  # APTOS duplicate groups (title row trimmed)

# 2. Training curves
hv, hc = pd.read_csv('results/history_vit.csv'), pd.read_csv('results/history_cnn.csv')
fig, ax = plt.subplots(1, 2, figsize=(FULL, 2.1))
for h, c, n in [(hv, VIT, 'ViT-B/16'), (hc, CNN, 'ResNet-50')]:
    ax[0].plot(h.epoch + 1, h.val_qwk_tta1, color=c, lw=2, marker='o', ms=4, label=n)
    ax[1].plot(h.epoch + 1, h.train_loss, color=c, lw=2, marker='o', ms=4, label=n)
    best = h.val_qwk_tta1.idxmax()
    ax[0].annotate(f'{h.val_qwk_tta1[best]:.3f}', (h.epoch[best] + 1, h.val_qwk_tta1[best]), textcoords='offset points',
                   xytext=(0, 5), ha='center', fontsize=7, color='#0b0b0b')
ax[0].set(xlabel='epoch', ylabel='val QWK (no TTA)', ylim=(0.5, 0.86))
ax[1].set(xlabel='epoch', ylabel='train loss (ordinal BCE)')
for a in ax:
    a.set_xticks(range(1, 11))
    a.legend(loc='lower right' if a is ax[0] else 'upper right')
fig.tight_layout()
fig.savefig(OUT / 'fig_curves.pdf')

# 3. QWK per variant and evaluation set
t = pd.read_csv('results/results_table.csv')
variants = ['vit', 'cnn', 'ensemble', 'vit+pair', 'cnn+pair', 'ensemble+pair']
labels = ['ViT', 'ResNet-50', 'Ensemble', 'ViT\n+pair', 'ResNet-50\n+pair', 'Ensemble\n+pair']
sets = [('val', 'Val', VIT), ('test_private', 'EyePACS Private test', CNN), ('external', 'Messidor-2 (external)', ENS)]
fig, ax = plt.subplots(figsize=(FULL, 2.2))
x = np.arange(len(variants))
w = 0.26
for i, (s, n, c) in enumerate(sets):
    v = [t[(t.variant == vv) & (t.set == s)].qwk.item() for vv in variants]
    bars = ax.bar(x + (i - 1) * w, v, w - 0.03, color=c, label=n, zorder=3)
    for b, val in zip(bars, v):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.002, f'{val:.3f}'[1:], ha='center', va='bottom', fontsize=6, color='#0b0b0b')
ax.set_xticks(x, labels)
ax.set_ylim(0.78, 0.872)
ax.set_ylabel('quadratic weighted kappa')
ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0))
ax.grid(axis='x', visible=False)
fig.tight_layout()
fig.savefig(OUT / 'fig_variants.pdf')

# 4. Confusion matrices of the final model (ensemble+pair); counts transcribed from 03_fusion_ensemble output
cm_private = np.array([[29135, 2005, 232, 25, 3], [1597, 1133, 305, 7, 0], [814, 1369, 2856, 1190, 52],
                       [16, 17, 186, 688, 70], [9, 33, 108, 338, 478]])
cm_ext = np.array([[836, 149, 31, 1, 0], [99, 118, 51, 2, 0], [12, 36, 236, 63, 0], [0, 0, 5, 56, 14], [0, 0, 2, 11, 22]])
assert cm_private.sum() == 42666 and cm_ext.sum() == 1744
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('blue', ['#ffffff', '#cde2fb', '#86b6ef', '#2a78d6', '#104281'])
fig, ax = plt.subplots(1, 2, figsize=(FULL, 3.0))
for a, cm, title in [(ax[0], cm_private, 'EyePACS Private test (n=42,666)'), (ax[1], cm_ext, 'Messidor-2 external (n=1,744)')]:
    r = cm / cm.sum(1, keepdims=True)
    a.imshow(r, cmap=cmap, vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            a.text(j, i, f'{r[i, j]:.2f}\n({cm[i, j]})', ha='center', va='center', fontsize=6.5,
                   color='white' if r[i, j] > 0.55 else '#0b0b0b')
    a.set(xticks=range(5), yticks=range(5), xlabel='predicted grade', ylabel='true grade', title=title)
    a.grid(False)
    a.spines[:].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / 'fig_cm.pdf')

# 5. Per-grade recall (classification reports of 02 runs and the final 03 variant; EyePACS = full 53,571 test)
recall = {
    'EyePACS test': {'ViT': [.955, .199, .516, .646, .533], 'ResNet-50': [.944, .275, .408, .694, .455],
                     'Ensemble+pair': [.928, .370, .461, .694, .507]},
    'Messidor-2': {'ViT': [.914, .222, .720, .707, .629], 'ResNet-50': [.813, .422, .608, .787, .629],
                   'Ensemble+pair': [.822, .437, .680, .747, .629]},
}
fig, ax = plt.subplots(1, 2, figsize=(FULL, 2.1), sharey=True)
for a, (name, d) in zip(ax, recall.items()):
    for i, ((m, v), c) in enumerate(zip(d.items(), [VIT, CNN, ENS])):
        a.bar(np.arange(5) + (i - 1) * w, v, w - 0.03, color=c, label=m, zorder=3)
    a.set_xticks(range(5), range(5))
    a.set_xlabel('true grade')
    a.set_title(name, pad=4)
    a.set_ylim(0, 1.05)
    a.grid(axis='x', visible=False)
ax[0].set_ylabel('recall (sensitivity) per grade')
fig.legend(*ax[0].get_legend_handles_labels(), ncol=3, loc='lower center', bbox_to_anchor=(0.5, 0.98))
fig.tight_layout()
fig.savefig(OUT / 'fig_recall.pdf')
print('figures written to', OUT)
