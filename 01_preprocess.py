# %% [markdown]
# # 01 · Preprocess EyePACS / APTOS / Messidor-2 into 768 px shards
#
# **Kaggle settings:** Accelerator **None** (CPU, uses no GPU quota). Internet not needed.
# Run with *Save Version → Save & Run All* so the output persists. Expected runtime is roughly 1–2 h, mostly I/O.
#
# **Inputs to attach** (edit `SRC` / `LABELS` below if your mount paths differ):
# | role | Kaggle source |
# |---|---|
# | EyePACS train + test images (original ~82 GB split zips) | competition `diabetic-retinopathy-detection` (accept its rules first) |
# | EyePACS 2015 **test labels** (released after the competition) | dataset `benjaminwarner/resized-2015-2019-blindness-detection-images` → `labels/testLabels15.csv` |
# | APTOS 2019 | competition `aptos2019-blindness-detection` |
# | Messidor-2 images | dataset `xyaustin/messidor2` |
# | Messidor-2 adjudicated grades (Krause et al. 2018) | dataset `google-brain/messidor2-dr-grades` |
#
# **What it does**
# 1. Streams every image straight out of the split zip archives. Nothing is extracted to disk, because 82 GB would not fit.
#    JPEGs are decoded at reduced scale through libjpeg's DCT scaling, which makes decoding about 4× faster.
# 2. Tight-crops the fundus disc, pads it to a centred square and resizes to **768 px**. Training runs at 512 px by default, where microaneurysms are about 2–4 px.
#    Keeping 768 px leaves room to train at 640–768 later. Ben Graham / CLAHE enhancement happens at load time in notebook 02, so it can be ablated.
# 3. Label-noise cleanup on APTOS. The dataset contains near-duplicate images, some with conflicting grades.
#    These are clustered by perceptual hash: groups with a conflicting grade are dropped, and only one copy of a consistent group is kept.
# 4. Makes a **patient-grouped** stratified train/val split, so the left and right eye of one patient never land on different sides.
#
# **Output** (`/kaggle/working`): `eyepacs_train.zip`, `eyepacs_test.zip`, `aptos.zip`, `messidor2.zip` (JPEG, stored uncompressed) and `labels.csv`.
# The images are sharded into zips because Kaggle notebook output has a ~500-file limit.

# %%
import bisect, functools, io, itertools, multiprocessing as mp, os, zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

INPUT = Path(os.environ.get('DR_INPUT', '/kaggle/input'))
OUT = Path(os.environ.get('DR_WORK', '/kaggle/working'))
SIZE = 768        # stored resolution
QUALITY = 95      # JPEG quality of stored shards (~10 GB total, under the 20 GB output limit)
DUP_BITS = 6      # max Hamming distance (of 256 bits) for two APTOS images to count as duplicates
VAL_FOLDS = 10    # val = 1/VAL_FOLDS of the labelled training pool
EXTS = {'.jpeg', '.jpg', '.png', '.tif', '.tiff'}

# Image source per shard: a directory, a .zip, or the first part (.001) of a split zip.
# If reading the original split zips fails, point the eyepacs entries at
# INPUT/'resized-2015-2019-blindness-detection-images/resized train 15' (and 'resized test 15'). Those images are already ~1024 px.
SRC = {
    'eyepacs_train': INPUT / 'diabetic-retinopathy-detection/train.zip.001',
    'eyepacs_test': INPUT / 'diabetic-retinopathy-detection/test.zip.001',
    'aptos': INPUT / 'aptos2019-blindness-detection/train_images',
    'messidor2': INPUT / 'messidor2/messidor-2/images',
}
LABELS = {
    'eyepacs_train': INPUT / 'diabetic-retinopathy-detection/trainLabels.csv.zip',
    'eyepacs_test': INPUT / 'resized-2015-2019-blindness-detection-images/labels/testLabels15.csv',
    'aptos': INPUT / 'aptos2019-blindness-detection/train.csv',
    'messidor2': INPUT / 'messidor2-dr-grades/messidor_data.csv',
}
SPLIT = {'eyepacs_train': 'train', 'aptos': 'train', 'eyepacs_test': 'test', 'messidor2': 'external'}
OUT.mkdir(parents=True, exist_ok=True)


