# Does Real-Only Modeling Improve Unseen-Generator Detection?

## A Controlled Study with Frozen DINOv2 Features

> **Key finding:** On the evaluated Tiny-GenImage derivative, a supervised probe trained with images from one known generator outperforms three real-only scoring methods on three held-out generators. Real-only methods produce more stable false-positive rates under JPEG recompression and blur, but detect very few synthetic images at the chosen operating point. The dataset itself also contains a severe stored-format shortcut. A supplementary test on content-matched images stored losslessly in both classes (B-Free extended Synthbuster, Section 5) gives the same ordering for the distance-based scores, while the magnitude score M0 reverses.

## Research questions

**Can modeling only real images improve detection of unseen synthetic image generators, compared with a binary classifier trained on just one synthetic generator?** Four related questions:

1. Does real-only scoring achieve higher absolute AUROC against held-out generators?
2. Do representation-magnitude and real-reference distances make different errors?
3. Can real-only scoring keep the real false-positive rate stable under JPEG and blur, without sacrificing useful detection sensitivity?
4. How do stored-file format and preprocessing choices affect the measurement?

All tests use frozen global representations; pixel-space reconstruction and patch-level localization are outside the scope.

## Study design

| Component                    | Protocol                                                                                                                                          |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dataset                      | [TheKernel01/Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage), fixed revision `89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69` |
| Encoder                      | Frozen DINOv2 ViT-S/14, with ViT-B/14 secondary check                                                                                             |
| Reference training           | `R_train`: 2,000 authentic images                                                                                                                 |
| Seen synthetic training      | `F_seen_train`: 1,986 Stable Diffusion 1.5 images, used only by B1                                                                                |
| Calibration                  | `R_cal`: 500 separate authentic images; nominal 5% FPR target                                                                                     |
| Evaluation                   | `R_test`: 500 authentic images; 300 each from SD15, ADM, GLIDE, Midjourney                                                                        |
| Unseen generators            | ADM, GLIDE and Midjourney (absent from B1 synthetic training)                                                                                     |
| Primary processing condition | D-ctrl: aligned native-resolution crop with JPEG-history harmonization                                                                            |
| Statistics                   | 2,000 shared, paired bootstrap resamples of test identities; 95% percentile intervals                                                             |

The source derivative has no SD 1.4 examples, despite listing SD14 in its schema; SD15 was substituted in the recorded protocol. “Unseen” refers only to the synthetic images used to train B1.

### Four detection scores

| ID     | Score                                                                           | Uses synthetic labels for fitting? |
| ------ | ------------------------------------------------------------------------------- | ---------------------------------- |
| **B1** | Logistic-regression decision score on L2-normalized post-LN DINOv2 CLS features | Yes, SD15                          |
| **O1** | Distance to the 5th-nearest real reference in normalized CLS space              | No                                 |
| **O2** | Ledoit–Wolf shrinkage Mahalanobis distance from real references                 | No                                 |
| **M0** | Negative pre-LayerNorm CLS L1 magnitude, fixed orientation                      | No                                 |

All thresholds are calibrated on the real-only `R_cal`; detection occurs when the score exceeds the fitted threshold.

## 1. Unseen-generator generalization

![Figure F1: per-generator AUROC and detection rates](report/figures/F1_e1_d_ctrl.svg)

*Figure F1. Left: AUROC by generator; right: true-positive rate at real-calibrated thresholds. Error bars: paired bootstrap 95% intervals.*

| Method | SD15 (seen) | ADM   | GLIDE | Midjourney | Mean unseen AUROC (95% CI)     |
| ------ | -----------:| -----:| -----:| ----------:| ------------------------------:|
| B1     | 0.955       | 0.670 | 0.888 | 0.930      | 0.829 [0.809, 0.850]           |
| O1     | 0.579       | 0.578 | 0.608 | 0.678      | 0.621 [0.591, 0.651]           |
| O2     | 0.592       | 0.575 | 0.639 | 0.638      | 0.617 [0.585, 0.646]           |
| M0     | 0.547       | 0.664 | 0.630 | 0.572      | 0.622 [0.592, 0.652]           |

At thresholds targeting a 5% FPR, unseen TPR is 52.7% for B1 but only 10.8%, 11.3% and 13.4% for O1, O2 and M0. The actual clean-test real FPR is 8.2% for B1, versus 4.8%, 5.2% and 6.0% for the real-only scores. M0 even scores higher on unseen generators than on SD15 (0.622 vs 0.547); with both values this low, the seen-to-unseen change says little about generalization.

The primary B1-minus-M0 mean unseen AUROC difference is 0.207 [0.169, 0.246]. With the secondary ViT-B/14 backbone, B1 still leads: mean unseen AUROC 0.845, versus 0.573 (O1), 0.613 (O2) and 0.727 (M0).

## 2. Benign post-processing

![Figure F2: FPR and TPR under JPEG and blur](report/figures/F2_e2_d_ctrl.svg)

*Figure F2. Clean-fitted models and thresholds are held fixed across JPEG quality 95, 75, 50 and Gaussian blur σ=0.5, 1.0. Both false-positive and true-positive responses are shown.*

| Method | Clean FPR | JPEG Q75 FPR | JPEG Q50 FPR | Blur 0.5 FPR | Blur 1.0 FPR |
| ------ | ---------:| ------------:| ------------:| ------------:| ------------:|
| B1     | 8.2%      | 21.8%        | 29.2%        | 17.2%        | 12.8%        |
| O1     | 4.8%      | 5.6%         | 7.2%         | 5.2%         | 5.4%         |
| O2     | 5.2%      | 6.8%         | 8.4%         | 6.2%         | 6.2%         |
| M0     | 6.0%      | 7.6%         | 7.2%         | 6.6%         | 6.8%         |

JPEG Q50 increases B1's real FPR by 21.0 percentage points (95% CI [17.6, 24.6]). The real-only scores move much less, but their detection sensitivity stays very low, so the comparison is a trade-off between stable false alarms and sensitivity.

## 3. Preprocessing and encoding shortcuts

![Figure F3: generator-specific preprocessing ladder](report/figures/F3_ladder.svg)

*Figure F3. For each generator, compares D-raw, D-jpeg and D-ctrl. D-native was added post hoc and appears only in the table below.*

| Method | D-raw unseen AUROC | D-jpeg | D-native (post hoc) | D-ctrl |
| ------ | ------------------:| ------:| -------------------:| ------:|
| B1     | 0.756              | 0.735  | 0.838               | 0.829  |
| O1     | 0.575              | 0.575  | 0.620               | 0.621  |
| O2     | 0.597              | 0.598  | 0.618               | 0.617  |
| M0     | 0.572              | 0.571  | 0.622               | 0.622  |

Switching from resized inputs to native crops changes B1 more than the mild JPEG Q96 harmonization does (+0.094 [0.080, 0.108] versus −0.020 [−0.027, −0.014]); native crops raise the real-only scores by 0.019 to 0.051. Native crops also change the field of view, so resampling and image content cannot be separated here.

Stored format alone reveals the label: every authentic image in the derivative is stored as JPEG and every synthetic image as PNG, and a metadata-only baseline reaches AUROC 1.000. Compression traces survive decoding, which is why D-jpeg and D-ctrl harmonize the JPEG history.

**Injected encoding cue.** The C0 model is the D-ctrl setup with both classes stored as lossless PNG. The C1 model is refitted and recalibrated with the real training and calibration images re-encoded as JPEG Q75, while the SD15 training images stay PNG. C1 is then tested with (a) real JPEG Q75 and synthetic PNG (cue kept), (b) both classes JPEG Q75, and (c) real PNG and synthetic JPEG Q75 (cue inverted).

![Figure F4: encoding intervention, AUROC, FPR, score shifts and TPR](report/figures/F4_e3c_injected_cue.svg)

*Figure F4. Panels: AUROC on fixed identities, real FPR at each condition's threshold, paired real-score shifts under the fixed C1 model, and TPR at each condition's threshold.*

The B1 mean unseen AUROC changes from 0.829 for C0 to 0.754, 0.701 and 0.708 for C1(a), C1(b) and C1(c). Part of this drop comes from refitting: with both test classes left as PNG, the C1 model already scores 0.070 [0.057, 0.082] below C0 (post hoc decomposition). Holding the C1 model fixed, JPEG-coding the synthetic test images (C1a → C1b) costs a further 0.053 [0.048, 0.059], while JPEG-coding only the real test images changes AUROC by −0.005 [−0.010, −0.001].

## 4. Reference reseeding (post hoc)

After the main experiment, `R_train` was redrawn with seeds 7, 13 and 29, in addition to the frozen seed 42. The seen synthetic training set, real calibration/test sets, and synthetic test identities were kept fixed. B1/O1/O2 were refitted; M0 does not depend on the reference set.

| Method | Seed 42 | Seed 7 | Seed 13 | Seed 29 | Range |
| ------ | -------:| ------:| -------:| -------:| -----:|
| B1     | 0.829   | 0.846  | 0.833   | 0.822   | 0.025 |
| O1     | 0.621   | 0.628  | 0.621   | 0.624   | 0.007 |
| O2     | 0.617   | 0.636  | 0.622   | 0.611   | 0.025 |
| M0     | 0.622   | 0.622  | 0.622   | 0.622   | 0.000 |

On every draw B1 exceeds each real-only score by at least 0.198 AUROC (seed 29, B1 minus O1, 95% CI [0.166, 0.230]), and no paired interval falls below 0.161. B1's clean real FPR stays between 8.2% and 8.8%. The draws overlap and share the calibration and test sets, so this checks sensitivity to the reference sample only.

Results: [AUROC](artifacts/tables/posthoc_reseed_auroc.csv) · [paired contrasts](artifacts/tables/posthoc_reseed_differences.csv) · [TPR/FPR](artifacts/tables/posthoc_reseed_operating_points.csv).

## 5. Content-matched test on lossless data (S-BF, supplementary)

In the Tiny-GenImage derivative every real image is JPEG and every synthetic image PNG (Section 3). To test whether the comparison depends on that split, the four methods were run unchanged on the [extended Synthbuster set released with B-Free](https://github.com/grip-unina/B-Free): 1,000 RAISE photographs and, for each, one FLUX and one Stable Diffusion 3.5 image generated from a caption of that photograph. All 3,000 files are 8-bit RGB PNG, and a quantization-lattice test found no JPEG history at the stored resolution in any class ([DATA_CARD_SUPP_BFREE.md](DATA_CARD_SUPP_BFREE.md)). B1 is trained on real versus SD3.5; FLUX is used only for testing.

| Component | S-BF protocol |
| --------- | ------------- |
| Source ids | 1,000, grouped by near-duplicates: 600 train, 200 calibration, 200 test; no id is shared between roles |
| Fitting | B1: real and SD3.5 images of the train ids; O1, O2: real train images only; M0: no fitting |
| Calibration | 200 real images, same 5% rule (k = 191) |
| Test | 200 real images, with the FLUX (unseen) and SD3.5 (seen by B1) images of the same ids |
| Colour | real images with an ICC profile converted to sRGB; sensitivity analysis on the 743 sRGB-tagged ids |
| Views | ViT-S/14 native 224 crop (primary); resize to 256 then crop; ViT-B/14 native crop |
| Statistics | 2,000 bootstrap resamples of test ids, with the real and synthetic images of an id drawn together |

![Figure F5: ROC curves for FLUX](report/figures/supp_bfree/sbf_roc_flux.svg)

*Figure F5. FLUX against the content-matched real test images, ViT-S/14 native crop. Left: all ids; right: sRGB-tagged ids only. Dotted line: 5% FPR.*

| Method | AUROC FLUX (95% CI) | TPR at 5% FPR | AUROC SD3.5 (seen by B1) | Calibrated real FPR |
| ------ | -------------------:| -------------:| ------------------------:| -------------------:|
| B1     | 0.734 [0.688, 0.779] | 21.0%        | 0.986                    | 5.0%                |
| O1     | 0.615 [0.582, 0.652] | 10.5%        | 0.638                    | 7.0%                |
| O2     | 0.655 [0.619, 0.692] | 7.5%         | 0.661                    | 8.5%                |
| M0     | 0.358 [0.311, 0.404] | 1.0%         | 0.426                    | 2.0%                |

TPR at 5% FPR uses a threshold set on the 200 real test scores, so all methods are compared at the same real FPR. On FLUX, each real-only score is below B1 in AUROC: −0.118 [−0.171, −0.063] for O1, −0.079 [−0.132, −0.024] for O2 and −0.376 [−0.441, −0.310] for M0. At 5% FPR, the intervals of the O1 and O2 differences from B1 (−10.5 and −13.5 percentage points) include 0 with 200 test pairs. B1 loses 0.252 [0.208, 0.297] AUROC from SD3.5 to FLUX, while O1 and O2 score the two generators alike. The sRGB-tagged subset (B1 0.772, O1 0.618, O2 0.666, M0 0.364), the resized view (B1 0.873, O1 0.593, O2 0.642, M0 0.328) and ViT-B/14 (B1 0.735, O1 0.639, O2 0.682, M0 0.418) give the same ordering.

O1 and O2 score about as well as on Tiny-GenImage (0.621 and 0.617 in D-ctrl), so their main results do not depend on the JPEG/PNG split. M0 does not transfer: FLUX and SD3.5 images have a larger pre-LayerNorm magnitude than the RAISE photographs, and its AUROC falls below 0.5. The M0 results of Sections 1 and 2 apply to Tiny-GenImage only. Resizing raises B1 on FLUX (+0.139 [0.094, 0.187]), the reverse of the D-raw versus D-ctrl ordering in Section 3.

<details>
<summary><strong>Figure F6: AUROC on FLUX for every S-BF cell</strong></summary>

![Figure F6: AUROC on FLUX by cell](report/figures/supp_bfree/sbf_auroc_cells.svg)

</details>

Results: [all ids](artifacts/tables/supp_bfree/final_main_all.csv) · [sRGB ids](artifacts/tables/supp_bfree/final_main_srgb.csv) · [real-only minus B1](artifacts/tables/supp_bfree/final_contrasts_all.csv) · [seen minus unseen](artifacts/tables/supp_bfree/final_seen_unseen_all.csv) · [cells](artifacts/tables/supp_bfree/final_cells_all.csv) · [comparison with Tiny-GenImage](artifacts/tables/supp_bfree/final_core_comparison.csv) · [diagnostics](artifacts/tables/supp_bfree/final_diagnostics.csv).

## Supplementary visual evidence

<details>
<summary><strong>S1: Score distributions for three data conditions</strong></summary>

**D-raw**

![S1 D-raw score distributions](report/figures/S1_score_distributions_d_raw.svg)

**D-jpeg**

![S1 D-jpeg score distributions](report/figures/S1_score_distributions_d_jpeg.svg)

**D-ctrl**

![S1 D-ctrl score distributions](report/figures/S1_score_distributions_d_ctrl.svg)

</details>

<details>
<summary><strong>S2: Per-image real-score changes under JPEG and blur</strong></summary>

The thin lines show within-image changes for the same held-out real image; the black line is their mean. Shifts are standardized by each method's clean real-test score standard deviation. Neither the model score nor threshold was standardized for evaluation.

**D-raw**

![S2 D-raw individual real-image responses](report/figures/S2_real_response_curves_d_raw.svg)

**D-ctrl**

![S2 D-ctrl individual real-image responses](report/figures/S2_real_response_curves_d_ctrl.svg)

</details>

## Code and reproduction

This repository contains the experiment code, configurations, frozen splits, saved predictions and figures of the study and of the S-BF supplement.

- [`src/`](src/): dataset auditing, duplicate grouping, deterministic partitioning, frozen DINOv2 feature extraction, B1/O1/O2/M0 scoring, calibration, E1 to E3 experiments, the two post hoc runs, analysis and plots. The S-BF supplement is in `src/supp_bfree_*.py` and reuses these modules unchanged.
- [`configs/`](configs/): four protocol configurations and pinned model/dataset revisions; `supp_bfree.yaml` for S-BF.
- [`tests/`](tests/): core checks for preprocessing, split integrity, label isolation and calibration.
- [`artifacts/predictions/`](artifacts/predictions/): compressed final and post hoc prediction records (E1, E2, E3, D-native, reseeding, S-BF), from which every table and figure is regenerated without model inference. Train-split pilot records are not included.
- [`artifacts/tables/`](artifacts/tables/): selected numerical summary CSVs.
- [`artifacts/manifest.csv`](artifacts/manifest.csv) and [`artifacts/splits.json`](artifacts/splits.json): frozen dataset identities, eligibility information and splits; [`artifacts/supp_bfree/`](artifacts/supp_bfree/): the S-BF file audit, manifest and splits.
- [`report/figures/`](report/figures/): the SVG figures shown in this README, generated by `src/figures.py` and, for S-BF, `src/supp_bfree_analysis.py` from the saved predictions.
- [`PROTOCOL.md`](PROTOCOL.md): the study protocol, with the post hoc analyses labeled and the S-BF protocol in Section 7.

### Recompute analysis and figures from stored predictions

For Python 3.11, in a dedicated environment:

```bash
python -m pip install -r requirements.lock
python -m pytest -q
python src/analysis.py
python src/figures.py
python src/supp_bfree_analysis.py
```

These commands regenerate tabular metrics and the figure set from saved predictions. They do not re-run DINOv2 inference or model fitting. Tests that need local view or feature caches, or the B-Free images, skip. On a fresh copy of this repository with the locked environment, `pytest` passes (109 passed, 13 skipped) and the three commands reproduce the committed tables and SVG figures byte for byte. A full raw-data rerun of the main study from this repository has not been performed; the S-BF experiment was rerun from its raw data ([REPRODUCIBILITY.md](REPRODUCIBILITY.md)).

### Full experiment reproduction

Obtain the pinned [Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage) Parquet data and official [DINOv2](https://github.com/facebookresearch/dinov2) source/weights, in the paths specified by [`configs/core.yaml`](configs/core.yaml). The E1 to E3 and post hoc entry points refuse to overwrite existing outputs, so a fresh run requires a separate disposable working copy, with the bundled reference predictions moved aside. [REPRODUCIBILITY.md](REPRODUCIBILITY.md) gives the commands and order for every result shown here, including D-native and the reseeding check, and states what has been verified.

## Limitations

- One primary frozen DINOv2 backbone; three held-out generators on Tiny-GenImage and one (FLUX, 200 test pairs) in S-BF.
- Only global class-token scoring is tested; pixel, patch, reconstruction and fine-tuned methods may behave differently.
- Bootstrap intervals resample the fixed test identities and hold the fitted models and thresholds fixed, so they omit training-set and dataset-level variation.
- D-native, the injected-cue decomposition and the `R_train` reseeding were added after the main results and are labeled post hoc. S-BF was also added afterwards, with its protocol fixed before any of its features were computed.
- In S-BF the real photographs were downscaled from camera resolution while the synthetic images are native 1024 × 1024 output; aspect ratios differ (3:2 versus 1:1), the content match is caption-level rather than pixel-level, and SD3.5 and FLUX are related latent generators.
- Dataset storage differences can make synthetic-image detection benchmark results misleading.

## Data, license, and citation

- **Code** (`src/`, `tests/`, `configs/`) and documentation: [MIT](LICENSE).
- **Dataset-derived files** (everything under `artifacts/` and `report/figures/` except the S-BF files, including the manifest with original file names and image hashes, the splits and the per-image prediction records): derived from [Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage) (CC BY-NC-SA 4.0), itself built from GenImage (Zhu et al., NeurIPS 2023). They are released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); see [LICENSE-DATA](LICENSE-DATA).
- **S-BF files** (`artifacts/supp_bfree/`, `artifacts/predictions/supp_bfree/`, `artifacts/tables/supp_bfree/`, `report/figures/supp_bfree/`, `DATA_CARD_SUPP_BFREE.md`): derived from the extended Synthbuster set of B-Free (F. Guillaro et al., "A Bias-Free Training Paradigm for More General AI-generated Image Detection", CVPR 2025), which builds on Synthbuster (Q. Bammey, IEEE Open Journal of Signal Processing, 2023) and on RAISE (D.-T. Dang-Nguyen, C. Pasquini, V. Conotter, G. Boato, "RAISE: A Raw Images Dataset for Digital Image Forensics", ACM MMSys 2015). For non-commercial research use under the terms of these sources; see [LICENSE-DATA](LICENSE-DATA).
- No dataset image is included. Attribution and pinned source details: [DATA_CARD.md](DATA_CARD.md) and [DATA_CARD_SUPP_BFREE.md](DATA_CARD_SUPP_BFREE.md).

Citation metadata: [CITATION.cff](CITATION.cff). Please cite it as an unpublished research project.
