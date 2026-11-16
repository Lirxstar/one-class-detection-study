# Reproducibility

This repository contains the Python experiment code, the protocol and configuration files, the frozen manifest and splits, the SVG figures, selected CSV tables and the compressed final and post hoc prediction records, for the main study, for the S-BF supplement (PROTOCOL.md Section 7) and for its post hoc analysis S-BF-EQ (Section 8). No dataset image bytes, pretrained model checkpoints, image views or feature arrays are included.

## Environment

Use Python 3.11, preferably in a new environment. The archived versions in `requirements.lock` reflect the original CPU/macOS arm64 setup and may not install identically on every platform.

```bash
python -m pip install -r requirements.lock
python -m pytest -q
```

Tests that need the local view or feature caches, or the B-Free images, skip until those exist. On a fresh copy of this repository with the locked environment: 121 passed, 14 skipped.

## Analysis from saved predictions

`artifacts/predictions/` holds the scores, thresholds and flags of E1, E2, E3 and of the two post hoc runs (D-native and `R_train` reseeding). To regenerate tables and figures without any model inference:

```bash
python src/analysis.py
python src/figures.py
```

`analysis.py` writes all tables to `artifacts/tables/` (a selection of them is committed); `figures.py` writes the nine main-study figures to `report/figures/` as SVG, plus PNG copies that `.gitignore` excludes. With the locked environment the committed tables and SVGs are reproduced byte for byte; other Matplotlib versions may change SVG bytes but not values.

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

## Supplementary experiment S-BF

### From saved predictions

```bash
python src/analysis.py
python src/supp_bfree_analysis.py
```

`supp_bfree_analysis.py` reads `artifacts/predictions/supp_bfree/` and three main-study tables that `analysis.py` writes (`table1_d_ctrl.csv`, `table1_d_raw.csv`, `posthoc_native_ladder.csv`). It writes the S-BF tables to `artifacts/tables/supp_bfree/` (a selection is committed) and the two figures to `report/figures/supp_bfree/`.

### Full rerun

1. Download the extended Synthbuster set listed in the [B-Free repository](https://github.com/grip-unina/B-Free) (section "Dataset of new generators") so that the folders are `data/BFree/real_RAISE_1k/`, `data/BFree/sd3_flux/sd3_large/` and `data/BFree/sd3_flux/flux/`. The SHA256 of every expected file is in `artifacts/supp_bfree/audit/files.csv`.
2. DINOv2 source and weights as in step 2 above.
3. In a disposable working copy, move `artifacts/predictions/supp_bfree` aside; `supp_bfree_run.py` refuses to overwrite it.

```bash
python src/supp_bfree_audit.py --workers 4
python src/supp_bfree_splits.py
python src/supp_bfree_features.py --workers 4
python src/supp_bfree_run.py
python src/supp_bfree_analysis.py
```

The audit rewrites `artifacts/supp_bfree/audit/` and `DATA_CARD_SUPP_BFREE.md`; `supp_bfree_splits.py` stops if the audit gate failed and otherwise rewrites `manifest.csv` and `splits.json`. The feature script asserts that the sRGB conversion leaves the 743 sRGB-tagged real images unchanged, that a second extraction of 32 views is bitwise identical, and, when the main-study view and feature caches exist, that 32 cached D-ctrl features are reproduced bitwise. `python src/supp_bfree_run.py --pilot` repeats the train-only development run; its records are not distributed.

A rerun should match the bundled records in identities, scores, thresholds, flags and selected C. `run_id`, `git_commit` and `supp_config_hash` identify the original run and differ; the published `configs/supp_bfree.yaml` differs from the original one only in comments.

## Post hoc analysis S-BF-EQ

### From saved predictions

```bash
python src/supp_bfree_eq_analysis.py
```

It reads `artifacts/predictions/supp_bfree_eq/` (B1 refitted on 600 images, five seeds), the O1, O2 and B1 records in `artifacts/predictions/supp_bfree/`, and `artifacts/supp_bfree_eq/subsamples.json`. It writes the tables to `artifacts/tables/supp_bfree_eq/` (a selection is committed) and Figure F7 to `report/figures/supp_bfree_eq/`.

### Rerun from the S-BF features

1. Build the S-BF feature cache (`artifacts/cache/supp_bfree/features/`) with the S-BF steps above, up to `supp_bfree_features.py`.
2. In a disposable working copy, move `artifacts/predictions/supp_bfree_eq` aside; `supp_bfree_eq_run.py` refuses to overwrite it.

```bash
python src/supp_bfree_eq_run.py
python src/supp_bfree_eq_analysis.py
```

The run checks every draw against `subsamples.json` before fitting. `python src/supp_bfree_eq_run.py --pilot` repeats the train-only development run and its code-path check; it needs the S-BF pilot records from `supp_bfree_run.py --pilot`. A rerun should match the bundled records in identities, scores, thresholds, flags and selected C; `run_id`, `git_commit` and the config hashes differ.

## What has been verified

- **Original runs.** Each final experiment (E1, E2, E3) and each post hoc run was executed once, after the protocol and splits were fixed and the test suite passed.
- **Independent working copy of the project.** Reruns from cached features and from raw data reproduced the E1 to E3 predictions. The D-native and reseeding runs were each executed once and have not been rerun from features or raw data.
- **This repository.** The test suite passes and `analysis.py` and `figures.py` reproduce the committed tables and figures from the saved predictions. No model fitting or inference of the main study has been rerun from this repository.
- **S-BF.** From a copy of this repository with the B-Free data, the audit, the splits, feature extraction and `supp_bfree_run.py` reproduced the bundled S-BF records (identities, scores, thresholds, flags and selected C), and `supp_bfree_analysis.py` reproduced the committed S-BF tables and SVG figures byte for byte.
- **S-BF-EQ.** From a copy of this repository with the S-BF feature cache, `supp_bfree_eq_run.py` reproduced the bundled S-BF-EQ records (identities, scores, thresholds, flags, selected C and CV tables), and `supp_bfree_eq_analysis.py` reproduced the committed S-BF-EQ tables and SVG figure byte for byte.

## Limits and licensing

Intervals are paired percentile bootstrap confidence intervals over fixed test identities, conditional on fitted models. The reseeding study does not represent multiple independent datasets. Consult `DATA_CARD.md` and `PROTOCOL.md` for data eligibility, leakage controls and interpretation. Code is MIT licensed (`LICENSE`); the dataset-derived files under `artifacts/` and `report/figures/` follow the Tiny-GenImage CC BY-NC-SA 4.0 license, and the S-BF and S-BF-EQ files the B-Free and RAISE terms (`LICENSE-DATA`).
