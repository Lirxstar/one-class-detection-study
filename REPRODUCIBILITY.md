# Reproducibility

This repository contains the Python experiment code, the protocol and configuration files, the frozen manifest and splits, nine SVG figures, selected CSV tables and the compressed final and post hoc prediction records. No dataset image bytes, pretrained model checkpoints, image views or feature arrays are included.

## Environment

Use Python 3.11, preferably in a new environment. The archived versions in `requirements.lock` reflect the original CPU/macOS arm64 setup and may not install identically on every platform.

```bash
python -m pip install -r requirements.lock
python -m pytest -q
```

Tests that need the local view or feature caches skip until those caches exist. On a fresh copy of this repository with the locked environment: 90 passed, 12 skipped.

## Analysis from saved predictions

`artifacts/predictions/` holds the scores, thresholds and flags of E1, E2, E3 and of the two post hoc runs (D-native and `R_train` reseeding). To regenerate tables and figures without any model inference:

```bash
python src/analysis.py
python src/figures.py
```

`analysis.py` writes all tables to `artifacts/tables/` (a selection of them is committed); `figures.py` writes the nine figures to `report/figures/` as SVG, plus PNG copies that `.gitignore` excludes. With the locked environment the committed tables and SVGs are reproduced byte for byte; other Matplotlib versions may change SVG bytes but not values.

## Full rerun from raw data

1. Download [TheKernel01/Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage) at commit `89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69` into `data/tiny_genimage/` (Parquet files under `data/tiny_genimage/data/`).
2. Clone [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) at commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8` into `models/dinov2/`. Download the official DINOv2 ViT-S/14 and ViT-B/14 weights into `checkpoints/` with the filenames and SHA256 hashes in `configs/core.yaml`.
3. Work in a disposable separate copy of the repository and move the bundled reference predictions out of it *before running*. The experiment entry points (`run_e1.py`, `run_e2.py`, `run_e3.py` and the two post hoc scripts) refuse to overwrite an existing prediction file.

```bash
# Run in a disposable working copy; keep the downloaded release unchanged.
mkdir -p ../reference-records
mv artifacts/predictions ../reference-records/predictions

# 1. Data audit, partitions, views and features
python src/audit_dataset.py --data-dir data/tiny_genimage --revision 89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69 --workers 4 --out artifacts/audit
python src/make_splits.py
python src/data_card.py
python src/make_conditions.py --workers 4
python src/extract_features.py --plan all
python src/dimension_check.py primary
python src/dimension_check.py secondary
python src/encoding_control.py

# 2. Main experiments (protocol fixed before validation scoring)
python src/run_e1.py
python src/run_e2.py
python src/run_e3.py

# 3. Post hoc analyses (only after E1 to E3)
python src/posthoc_native.py
python src/posthoc_reseed.py

# 4. Tables and figures
python src/analysis.py
python src/figures.py
```

The order matters for the two post hoc runs:

| Script | Result in README | Needs | Writes |
| --- | --- | --- | --- |
| `src/posthoc_native.py` | D-native column (Section 3), PROTOCOL Section 5 (post hoc) | the Parquet data and the view cache from `make_conditions.py` (`artifacts/cache/views/index.csv`, `d_ctrl.npy`) | `d_native.npy` views (asserting that real D-native views equal the D-ctrl views), ViT-S/14 features for the study partitions, then the E1 protocol: `artifacts/predictions/posthoc/posthoc_primary_d_native.csv.gz`, `posthoc_diagnostics.json` |
| `src/posthoc_reseed.py` | Reference reseeding (Section 4), PROTOCOL Section 5 (post hoc) | the D-ctrl views and clean ViT-S/14 features from step 1 and the E1 D-ctrl predictions from step 2 of this working copy | after checking that seed 42 reproduces `R_train`, that views and features of 64 `R_train` images are reproduced bitwise, and that seed 42 through the reseeding path reproduces the E1 D-ctrl predictions exactly: views and features for 3,202 additional real train images, refits for seeds 7, 13 and 29, `posthoc_reseed_primary_d_ctrl.csv.gz`, `posthoc_reseed_diagnostics.json` |

The original reseeding run also compared its seed-42 path with the train-only pilot predictions. Those pilot records are not distributed, so that comparison is skipped here and `python src/posthoc_reseed.py --dev-check`, which needs them, is not available.

To check a rerun, compare each new file in `artifacts/predictions/` with the same file in `../reference-records/predictions/`: identities, scores, thresholds and flags should match; `run_id` and `git_commit` differ by design.

## What has been verified

- **Original runs.** Each final experiment (E1, E2, E3) and each post hoc run was executed once, after the protocol and splits were fixed and the test suite passed.
- **Independent working copy of the project.** Reruns from cached features and from raw data reproduced the E1 to E3 predictions. The D-native and reseeding runs were each executed once and have not been rerun from features or raw data.
- **This repository.** The test suite passes and `analysis.py` and `figures.py` reproduce the committed tables and figures from the saved predictions. No model fitting or inference has been rerun from this repository.

## Limits and licensing

Intervals are paired percentile bootstrap confidence intervals over fixed test identities, conditional on fitted models. The reseeding study does not represent multiple independent datasets. Consult `DATA_CARD.md` and `PROTOCOL.md` for data eligibility, leakage controls and interpretation. Code is MIT licensed (`LICENSE`); the dataset-derived files under `artifacts/` and `report/figures/` follow the dataset's CC BY-NC-SA 4.0 license (`LICENSE-DATA`).
