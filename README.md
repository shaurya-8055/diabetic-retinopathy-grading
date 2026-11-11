# Diabetic Retinopathy Severity Grading for Mass Screening

This project grades retinal fundus photographs into the 5 ICDR levels: 0 none, 1 mild, 2 moderate, 3 severe NPDR, 4 PDR.
It compares an ordinal ViT-B/16 with a ResNet-50 and ensembles them with paired-eye fusion.
It reports per-grade confusion matrices, validates externally on Messidor-2, and ships a `screen()` tool with a clinical-limitations section.
Everything runs on the **Kaggle free tier** (2×T4, 12 h sessions, 20 GB output).

Github Url = [`https://github.com/shaurya-8055/diabetic-retinopathy-grading`](https://github.com/shaurya-8055/diabetic-retinopathy-grading)

**Deliverables:**
- 2-page proposal: [`docs/proposal/proposal.pdf`](docs/proposal/proposal.pdf)
- IEEE report (9 pages): [`docs/report/report.pdf`](docs/report/report.pdf)

## Results

Every cut-off and fusion weight is fitted on val only. The EyePACS Private test has 42,666 images and Messidor-2 has 1,744, the latter with adjudicated grades and never used for training or tuning.

| Model | Val QWK | EyePACS Private QWK | rDR AUC | Messidor-2 QWK | rDR AUC |
|---|---|---|---|---|---|
| ViT-B/16 @512 | 0.843 | 0.808 | 0.937 | 0.840 | 0.964 |
| ResNet-50 @512 | 0.834 | 0.803 | 0.937 | 0.819 | 0.958 |
| Ensemble | 0.852 | 0.818 | 0.947 | **0.853** | **0.969** |
| ViT + paired-eye fusion | 0.860 | 0.830 | 0.950 | 0.831 | 0.964 |
| **Ensemble + paired-eye fusion (final)** | **0.864** | **0.838** | **0.955** | 0.843 | **0.969** |

At the screening cut-off (90 % referable-DR sensitivity on val), the final model gives:

| Set | Sensitivity | Specificity |
|---|---|---|
| EyePACS Private | 86.5 % | 93.2 % |
| Messidor-2 | 95.6 % | 79.1 % |

The full table is in [`results/results_table.csv`](results/results_table.csv), and the per-epoch training curves are in `results/history_*.csv`.

## Repository layout

| path | what |
|---|---|
| `01_preprocess.py` / `.ipynb` | **CPU** notebook. Streams the original 82 GB EyePACS split zips in place, crops the fundus to 768 px, removes conflicting APTOS duplicates (perceptual hash), and makes a patient-grouped train/val split. Writes ~10 GB of zip shards plus `labels.csv`. |
| `02_train_eval.py` / `.ipynb` | **GPU T4×2** notebook. Trains the ordinal ViT or CNN with DDP, runs 4-view TTA evaluation (val, EyePACS test, Messidor-2) with bootstrap CIs and confusion matrices, and holds the `screen()` tool and clinical limitations. Run it once per architecture. |
| `03_fusion_ensemble.py` / `.ipynb` | **CPU** notebook. Combines the `pred_*.csv` files of the 02 runs: ensemble, paired-eye fusion, screening cut-off, Private-test split. Picks the final variant by val QWK. |
| `smoke_test.py` | End-to-end CPU test of all three notebooks on synthetic data shaped like the Kaggle inputs. It exercises the 2-process DDP path. |
| `build_notebooks.py` | Regenerates the `.ipynb` files from the `.py` sources (`# %%` cells), which are the source of truth. |
| `results/` | Final metrics table and training histories, extracted from the executed notebooks. |
| `Executed-Notebooks/`, `Executed-Files-v3/` | The executed Kaggle notebooks and logs that produced every number in the report. |
| `docs/` | Proposal and report (LaTeX + PDF). `docs/report/make_figures.py` rebuilds every figure; `docs/meta.tex` holds the institute, course and repository URL. |

## Running on Kaggle

1. **01_preprocess**: *File → Import Notebook*, then set Accelerator to **None**. *Add Input*:
   - competition `diabetic-retinopathy-detection`; join it and accept the rules.
   - competition `aptos2019-blindness-detection`.
   - datasets `benjaminwarner/resized-2015-2019-blindness-detection-images`, needed only for `labels/testLabels15.csv`, the EyePACS test labels.
   - datasets `xyaustin/messidor2` (images) and `google-brain/messidor2-dr-grades` (adjudicated grades).

   Check that the `SRC`/`LABELS` paths match `/kaggle/input`, then *Save Version → Save & Run All*. This takes about 1–2 h of CPU time.
2. **02_train_eval**: set Accelerator to **GPU T4×2** and Internet to **On**. *Add Input → Notebooks →* `01_preprocess`, then *Save & Run All*.
   - For the CNN, set `MODEL = 'resnet50.a1_in1k'` and `LR = 1e-4`.
   - In an interactive session use **Restart & Run All**: the per-GPU processes are forked, which only works before the kernel has touched CUDA.
   - To re-evaluate without training, set `CKPT` to a saved `model.pt`.
3. **03_fusion_ensemble**: set Accelerator to **None**. Add both 02 runs and the `benjaminwarner` dataset as inputs, then *Save & Run All*.

**Note on multi-GPU.** The reported runs used `nn.DataParallel`, which is numerically fine but slow:
- ResNet-50 trained at only 16 img/s, and host RAM leaked until the kernel died.
- The cause: `DataParallel` starts new threads on every forward pass, while PyTorch caches cuDNN plans per thread.

The current `02` uses DistributedDataParallel instead (see report Sec. IV-H). That version is verified by `smoke_test.py` on CPU. It has not yet been timed on Kaggle GPUs.

## Local checks

```bash
pip install -r requirements.txt
python smoke_test.py                      # CPU-only, ~1 min, synthetic data
python docs/report/make_figures.py        # rebuild report figures from the executed notebooks
cd docs/report && tectonic report.tex     # or upload docs/ to Overleaf; same for docs/proposal/proposal.tex
```

## Team

Abhinav Shukla (230101002) · Amit Kumar (230101021) · Shaurya Shakya (230101119) · Nitin Gupta (230101083) · Utkarsh Kumar (230101147) · Amit Usara (230101023) · Somil Gupta (230101135) · Krishna Kumar Gupta (230101068) · Saurabh Mishra (230101116) · Shreyansh Anand (230101128)

The work split is in the proposal (§5) and the report (Team Contributions).
