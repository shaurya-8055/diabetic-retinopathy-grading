# Diabetic Retinopathy Severity Grading: Kaggle pipeline

This pipeline grades fundus photos into 5 ICDR levels (0 none, 1 mild, 2 moderate, 3 severe NPDR, 4 PDR) to support screening camps.
It produces per-grade confusion matrices, QWK with bootstrap CIs, referable-DR sensitivity and specificity, external validation on Messidor-2, a `screen()` tool, and a clinical-limitations section.

| file | what |
|---|---|
| `01_preprocess.ipynb` | **CPU** notebook. Reads the original 82 GB EyePACS split zips in place, crops the fundus to 768 px, deduplicates APTOS, and makes a patient-grouped train/val split. Writes ~10 GB of sharded zips plus `labels.csv`. |
| `02_train_eval.ipynb` | **GPU T4 x2** notebook. Trains the ordinal ViT/CNN, evaluates with TTA (val, EyePACS test, Messidor-2), and holds the screening tool and clinical limitations. Run it once per architecture or ablation. |
| `03_fusion_ensemble.ipynb` | **CPU** notebook. Combines the prediction CSVs of any number of 02 runs: ensemble, paired-eye fusion, and a screening operating point at 90 % rDR sensitivity. Writes `results_table.csv` and picks the final variant by val QWK. |
| `*.py` | Source of truth (`# %%` cells). Run `python build_notebooks.py` to regenerate the `.ipynb` files after editing. |
| `smoke_test.py` | CPU test of both stages on synthetic data shaped like the Kaggle inputs (split zips, csv.zip, duplicates, a blank image). |

## Running on Kaggle

1. **01_preprocess**: *File → Import Notebook*. Set Accelerator to **None**. *Add Input*:
   - competition `diabetic-retinopathy-detection` (join it and accept the rules; late joining is allowed)
   - dataset `benjaminwarner/resized-2015-2019-blindness-detection-images`, needed only for `labels/testLabels15.csv`, the EyePACS test labels released after the competition
   - competition `aptos2019-blindness-detection`
   - dataset `xyaustin/messidor2` for the images, and `google-brain/messidor2-dr-grades` for the adjudicated grades (the latter holds only the CSV)

   Check that the paths in the `SRC`/`LABELS` cell match `/kaggle/input`, then **Save Version → Save & Run All**.
   Any source that is missing is skipped with a message rather than crashing.
2. **02_train_eval**: import it, then set Accelerator to **GPU T4 x2** and Internet to **On**. *Add Input → Notebooks →* your `01_preprocess`, then **Save & Run All**.
   Training and evaluation run as one forked process per GPU (DistributedDataParallel). In an interactive session, use **Restart & Run All**, because the GPU processes can only be forked from a kernel that hasn't touched CUDA yet.
   The notebook stops with a clear message if CUDA is already initialised.
   `nn.DataParallel` is not used: on Kaggle's PyTorch it made CNNs about 10× slower and leaked host RAM until the kernel died.
   To re-evaluate without retraining, add the run's output as input and set `CKPT = '/kaggle/input/.../model.pt'`.
   For the CNN, set `MODEL = 'resnet50.a1_in1k'` and `LR = 1e-4`.
3. **03_fusion_ensemble**: set Accelerator to **None**. *Add Input → Notebooks →* every 02 run you want to combine, plus `benjaminwarner/resized-2015-2019-blindness-detection-images` for the Private-test split. Then **Save & Run All**.

### Budget (Kaggle free tier: 30 GPU-h/week, 12 h/session, 20 GB output, ~500 output files, 4 CPUs)
All of these numbers are **estimates**. Nothing has been run on Kaggle yet.
| step | hardware | est. time |
|---|---|---|
| 01 preprocess (88k + 5k images) | CPU, 0 GPU-h | ~1–2 h, I/O-bound |
| 02 ViT-B/16 @512, 10 epochs | T4 x2 | ~4.5–5.5 h measured under DataParallel (24 img/s); DDP should be somewhat faster. The logs print img/s, and `TIME_BUDGET_H` stops training early |
| 02 evaluation, 4×TTA on ~60k images | T4 x2 | ~40 min |

That leaves weekly quota for a second model (e.g. the EfficientNet-B4 CNN baseline) or a `PREPROC` ablation.

## Design decisions and the literature behind them

- **Resolution.** Microaneurysms span only a few pixels even in the originals. Krause et al. 2018 found that moving from 299 to 779 px input improved grading. Kaggle-2015 winners (Graham; o_O) trained at ≥512 px.
  So the shards are stored at **768 px** and training defaults to **512 px**. For ViT-B/16 that is 1024 tokens, with position embeddings interpolated from the 384 checkpoint (as Dosovitskiy et al. 2021 do when fine-tuning at higher resolution).
  If GPU time allows, try `IMG = 640` with `GRAD_CKPT = True`.
- **Huang et al. 2021** ran a controlled ablation of a ResNet-50 on this exact EyePACS split, reporting test kappa on the Private 42,670 images.
  They found the following:
  - Raising the input from 512 to 768 px gained +0.010.
  - MSE regression beat cross-entropy by +0.020. Our cumulative-ordinal head with an expected-grade score plays the same role.
  - Graham and CLAHE enhancement each **cost** about 0.005.
  - Natural (instance-balanced) sampling beat class-balanced sampling, which overfit the rare grades.
  - Paired-eye feature fusion was the biggest late gain, +0.027, taking them to 0.858.
  - A 10-model ensemble added +0.005, for a final 0.863.

  Our `PREPROC` and `SAMPLE_ALPHA` defaults go against the first two of their findings, so both are ablations to run.
- **Ben Graham preprocessing** (2015 winner): `4·img − 4·GaussianBlur(img, σ=size/30) + 128`, then clip to 90 % of the radius. It removes illumination and camera colour differences while keeping lesion edges.
  That should help most on cross-dataset (Messidor-2) transfer. It runs at load time, so `PREPROC = 'clahe' | 'none'` is a one-line ablation.
- **Ordinal formulation.** The head predicts the 4 cumulative probabilities P(y > k), trained with BCE (Frank & Hall 2001; CORAL, Cao et al. 2020).
  The expected grade Σ P(y > k) is a continuous score. Cut-offs are tuned on val for QWK, the APTOS-2019 top-solution trick of regression plus an optimised rounder.
  Unlike softmax, this ordering penalises an error of 0→4 more than 0→1, matching what the quadratic-weighted kappa metric rewards.
- **Imbalance.** EyePACS is about 73 % grade 0 and about 2 % grade 4. Square-root class-rebalanced sampling (`SAMPLE_ALPHA = 0.5`) is a middle ground.
  Natural sampling ignores the rare grades, and full balancing over-predicts them. Threshold tuning on the naturally distributed val set then recalibrates.
  Every report prints per-grade recall and true-vs-predicted grade distributions, plus per-source QWK, so bias towards grade 0 or 2 is visible.
- **Label noise.** EyePACS has one grader per image, and inter-grader agreement on DR is moderate (Krause 2018). The pipeline uses ordinal label smoothing, drops the APTOS duplicates whose grades conflict, and judges final quality on the **adjudicated** Messidor-2 grades.
  A next step, if time allows, is confident-learning style filtering of high-loss training samples.
- **Leakage.** The val split is grouped by patient, so a patient's left and right eye never land on different sides. APTOS duplicates are grouped as well.
- **TTA.** Four flip/rotate views are averaged. Their standard deviation flags the 5 % least stable images for human review.
- **External validation.** Messidor-2 (France, 1,744 gradable images, 3-retina-specialist adjudication) is never used for tuning.
  Voets et al. 2019 could not reproduce Gulshan et al. 2016's Messidor-2 results with public data, reporting an rDR AUC of about 0.85 against 0.99, so expect a domain-shift drop and report it honestly.

### Possible extensions (not built)
- **Ensemble.** Run 02 twice, once with the ViT and once with `tf_efficientnet_b4.ns_jft_in1k`, and average the `score` columns in the `pred_*.csv` files.
- **RETFound** (Zhou et al., Nature 2023) is a retina-pretrained ViT-L. Its weights are gated on HuggingFace, and ViT-L at 512 px is heavy on a T4.
- **Both-eye features.** Graham's winning solution combined features from the left and right eye.
- **Lesion-aware attention** (Sun et al., CVPR 2021).
- **Grad-CAM / attention maps** to make the screening output explainable.

## References
- Gulshan V. et al. *Development and validation of a deep learning algorithm for detection of DR in retinal fundus photographs.* JAMA 2016.
- Krause J. et al. *Grader variability and the importance of reference standards for evaluating ML models for DR.* Ophthalmology 2018.
- Graham B. *Kaggle Diabetic Retinopathy Detection competition report.* 2015.
- Huang Y. et al. *Identifying the key components in ResNet-50 for diabetic retinopathy grading from fundus images: a systematic investigation.* arXiv:2110.14160, 2021.
- Voets M., Møllersen K., Bongo L.A. *Reproduction study using public data of: Development and validation of a deep learning algorithm…* PLoS ONE 2019.
- Cao W., Mirjalili V., Raschka S. *Rank consistent ordinal regression for neural networks (CORAL).* Pattern Recognition Letters 2020.
- Dosovitskiy A. et al. *An Image is Worth 16x16 Words (ViT).* ICLR 2021.
- Sun R. et al. *Lesion-Aware Transformers for Diabetic Retinopathy Grading.* CVPR 2021.
- Zhou Y. et al. *A foundation model for generalizable disease detection from retinal images (RETFound).* Nature 2023.
- Abràmoff M.D. et al. *Pivotal trial of an autonomous AI-based diagnostic system for detection of DR in primary care offices (IDx-DR).* npj Digital Medicine 2018.
- Beede E. et al. *A Human-Centered Evaluation of a Deep Learning System Deployed in Clinics for the Detection of DR.* CHI 2020.
