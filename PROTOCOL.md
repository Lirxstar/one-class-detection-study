# Experimental Protocol

**Study:** Real-only synthetic-image detection with frozen DINOv2 features  
**Protocol version:** v0.6.6  
**Supplementary experiment:** S-BF 1.0 (Section 7)  
**Primary condition:** D-ctrl  
**Machine-readable settings:** [`configs/core.yaml`](configs/core.yaml), [`configs/conditions.yaml`](configs/conditions.yaml), [`configs/e2_postprocessing.yaml`](configs/e2_postprocessing.yaml), [`configs/e3_encoding.yaml`](configs/e3_encoding.yaml); S-BF: [`configs/supp_bfree.yaml`](configs/supp_bfree.yaml)

The core E1–E3 experiments were specified before validation scoring. D-native, E3C decomposition, and reference reseeding were conducted as post-hoc analyses. Exact experimental settings are provided in the YAML configuration files.

## 1. Research questions

1. **Unseen-generator detection:** Do real-only methods generalize better than a probe trained on one synthetic generator, in *absolute* AUROC and sensitivity?
2. **Magnitude versus geometry:** How do class-token magnitude, kNN distance and Mahalanobis distance compare, including overlap of real-image false alarms?
3. **Benign transformations:** How do JPEG recompression and blur affect false-positive rates and synthetic-image detection at fixed thresholds?
4. **Dataset shortcuts:** How strongly do stored-file metadata, preprocessing and an injected JPEG cue affect the evaluation?

Headline comparisons use **D-ctrl**. D-raw and D-jpeg are controls. A smaller seen-to-unseen gap alone is not evidence of better detection.

## 2. Dataset, eligibility and splits

**Source:** `TheKernel01/Tiny-GenImage`, revision `89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69`; upstream identified by its dataset card as `yangsangtai/tiny-genimage`. The dataset card specifies **CC BY-NC-SA 4.0**. Stored bytes are audited with `pyarrow`, not decoded implicitly by the Hugging Face `datasets` loader. Each row is identified as `<split>/<parquet stem>/<row index>`; Parquet SHA256 checksums are recorded in `artifacts/provenance/`.

The derivative contains **no SD14 examples**. Accordingly, **SD15** is the seen generator for B1; **ADM, GLIDE and Midjourney** are held out from B1's synthetic training. `VQDM` and `Wukong` were used only for train-split development; `BigGAN` had no eligible samples. “Unseen” does **not** mean unseen by DINOv2 pretraining.

### Eligibility and leakage control

- **Real:** stored JPEG with *exact* IJG quality-96 luma/chroma quantization tables; RGB, three JPEG components, 4:4:4 subsampling, absent/1 EXIF orientation, and both sides at least 224 px. No EXIF transpose is applied.
- **Synthetic:** stored PNG, at least 224 px on both sides; RGB or fully opaque RGBA (alpha = 255 at every pixel, checked before RGB conversion). ADM images are RGBA PNG in this derivative.
- **Duplicates:** exact decoded-RGB SHA1 plus 64-bit dHash (9 × 8 bilinear grayscale, Hamming distance ≤ 4). Union-find groups are computed across all 35,000 rows. Groups spanning source splits or authenticity labels are excluded; otherwise, only the lexicographically smallest `row_id` in each group is eligible.
- **Selection:** within each eligible role/generator pool, sort by `row_id` then apply a fresh `numpy.random.default_rng(42).permutation`; no label-based stratification. Train and validation partitions are disjoint and never substituted for one another.

| Partition       | Source                               | Final size        | Purpose                                          |
| --------------- | ------------------------------------ | -----------------:| ------------------------------------------------ |
| `R_train`       | Train, real                          | 2,000             | Fit real-only references and the real half of B1 |
| `F_seen_train`  | Train, SD15                          | 1,986             | Fit supervised B1 and metadata baseline only     |
| `R_cal`         | Validation, real                     | 500               | Set clean real-only thresholds                   |
| `R_test`        | Validation, real                     | 500               | Measure real FPR and provide AUROC negatives     |
| `F_seen_test`   | Validation, SD15                     | 300               | Seen-generator evaluation                        |
| `F_unseen_test` | Validation, ADM / GLIDE / Midjourney | 300 per generator | Unseen-generator evaluation                      |

The SD15 training count falls below its 2,000 target because only 1,986 eligible rows remain. Other final targets were met. Historical development-only partitions (`P_cal`, `P_test`, `P_fake`) use unused **training-split** real images and development generators, never final validation data. Frozen identities are in `artifacts/manifest.csv` and `artifacts/splits.json`.

## 3. Image conditions and encoder

All views are **224 × 224 RGB uint8**, decoded using Pillow. Real images preserve their existing JPEG history; synthetic views are optionally JPEG-harmonized at native resolution.

| Condition    | Processing                                                                                                                        | Role                        |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| **D-raw**    | Resize short side to 256 (bicubic, long side floored), then 224 center crop                                                       | Baseline                    |
| **D-jpeg**   | JPEG Q96 round trip on synthetic images at native size, then D-raw resize/crop                                                    | Compression-history control |
| **D-ctrl**   | Same synthetic JPEG Q96 harmonization, **no resize**, 224 native crop with top/left center offsets rounded down to multiples of 8 | **Primary condition**       |
| **D-native** | D-ctrl crop without synthetic JPEG harmonization                                                                                  | **Post-hoc** control        |

JPEG uses 4:4:4 subsampling (Pillow code 0), `optimize=False`, `progressive=False` and standard IJG tables. The D-ctrl offset-multiple rule is 16 for modal real 4:2:0, otherwise 8; the audited modal real format was 4:4:4, so **m = 8**. D-jpeg versus D-ctrl changes both spatial resampling and field of view; it is **not** a pure resizing intervention. Re-encoding also cannot erase earlier compression history.

**Encoder:** frozen DINOv2 `dinov2_vits14` without registers, source commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`, checkpoint SHA256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`. Secondary E1 check: `dinov2_vitb14`, SHA256 `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`. Models run in `eval()` under `no_grad()` on CPU/float32 (batch size 32, six Torch threads), using ImageNet normalization.

From one `forward_features` call, retain `h_pre = x_prenorm[:, 0, :]` and `h_post = x_norm_clstoken`; normalize the latter to `z = h_post / max(||h_post||₂, 1e-12)`. The primary small-model feature width is 384. A 64-image training-data check requires bitwise-identical repeated extractions. Dimension-dominance diagnostics use `R_train` only; exploratory **M0-x** removes dimensions with mean absolute pre-LN activation above 10× the median, without changing M0.

## 4. Methods and calibration

All scores are float64, and **higher means more likely synthetic**.

| Method | Score / fitting rule                                                                                                                                                                                                             | Synthetic labels in fitting? |
| ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| **B1** | L2 logistic-regression decision function on `z`; `lbfgs`, `max_iter=5000`, no class weighting; 5-fold `StratifiedGroupKFold` over train rows only, C ∈ {0.01, 0.1, 1, 10, 100}, choose highest mean fold AUROC (ties: smaller C) | **Yes:** SD15 `F_seen_train` |
| **O1** | Cosine distance to the **5th** nearest `R_train` reference in `z` space                                                                                                                                                          | No                           |
| **O2** | Shrinkage Mahalanobis distance using `LedoitWolf(assume_centered=False)` on `z(R_train)`, solved with `numpy.linalg.solve`                                                                                                       | No                           |
| **M0** | Negative L1 norm of the pre-LayerNorm class token, `−‖h_pre‖₁`; no fitting, orientation fixed and never flipped                                                                                                                  | No                           |

Exploratory checks use B1 with C=1, O1 with k=1/10, and M0 L2, L1/L2 and dominant-dimension variants. These do not replace the four main methods.

**Metadata baseline (E3A):** stored format, width/height, aspect ratio, bytes per pixel, estimated JPEG quality and JPEG indicator; `StandardScaler` + logistic regression. **The scaler is fitted independently inside each grouped CV training fold**, then fitted on all allowed training rows after C selection. The primary baseline fits real versus SD15; the supplementary baseline uses eligible train rows from other generators too. It measures **stored-file confounding**, not what the visual models necessarily exploit.

**Threshold:** for each method, condition, encoding and backbone, use only **clean real `R_cal`** scores. With n=500, choose the k-th smallest calibration score, `k = ceil(0.95(n+1)) = 476`; classify as synthetic only when `score > threshold` (ties are not flagged). Thresholds are reused unchanged during E2 transformations and among test variants of the same E3C fit. A nominal **5% FPR target** does not guarantee 5% realized test FPR.

## 5. Experiments

- **E0 — Dataset audit:** inspect encoded bytes, labels, JPEG properties, dimensions, duplicate groups and eligibility **before validation scoring**. The audit found no SD14 rows and permitted opaque RGBA ADM images and 4:4:4 real JPEGs; these choices were resolved before final evaluation.
- **E1 — Generator generalization:** fit B1/O1/O2 and score M0; calibrate on `R_cal`; evaluate the shared `R_test` against each synthetic generator. Report per-generator AUROC/TPR/FPR, and the equal-weight mean across ADM, GLIDE and Midjourney. The secondary ViT-B/14 check is E1, D-ctrl only. Compare real-image false-alarm overlap between methods.
- **E2 — Benign post-processing:** apply JPEG **Q95, Q75, Q50** and Gaussian blur **σ=0.5, 1.0** to D-ctrl and D-raw test views. JPEG uses the same 4:4:4 settings; blur uses `scipy.ndimage.gaussian_filter` with reflect mode, truncate=4.0 and round/clip to uint8. Models and clean thresholds remain **frozen**, with no refitting or recalibration.
- **E3A — Metadata shortcut:** test the metadata-only logistic baseline per generator.
- **E3B — Processing ladder:** compare D-raw, D-jpeg and D-ctrl on the same test identities; report method ranks and per-generator changes.
- **E3C — Injected encoding cue:** on D-ctrl views, C0 applies lossless PNG to both classes; its view equality is checked bitwise. C1 fits real train/calibration as JPEG Q75 and seen synthetic train as PNG. One **C1** model and threshold are evaluated with (a) real JPEG / synthetic PNG, (b) both JPEG, and (c) real PNG / synthetic JPEG. C0 and C1 are **refitted and recalibrated separately**; comparisons within C1 isolate test-encoding changes more directly.

**Post-hoc analyses (after E1–E3 results):** D-native, C1 encoding decomposition, descriptive error/score diagnostics and a reference-sampling study. For reseeding, redraw only `R_train` with seeds **7, 13, 29** (seed **42** is the original), keeping `F_seen_train`, `R_cal`, `R_test` and all synthetic test identities fixed. Refit B1/O1/O2, recalibrate on the fixed `R_cal`, and leave M0 unchanged. The four draws are **not independent dataset replications**.

## 6. Metrics, integrity and interpretation

- **Metrics:** AUROC (synthetic positive), real FPR, generator-specific TPR and counts; average precision is interpreted only at its evaluated prevalence. Do not substitute accuracy, F1 or ECE as headline metrics.
- **Uncertainty:** 2,000 **paired percentile-bootstrap** draws, 95% intervals, seed **2026**. Real and each synthetic generator are sampled independently per draw, while matched methods, conditions and encoding variants share each identity's draws. Thresholds are **not resampled**. Intervals are conditional on the trained models, calibration and fixed dataset.
- **Leakage control:** B1 uses only real/SD15 **training** labels; O1/O2/M0 receive no synthetic labels. No validation labels or scores select hyperparameters, reference sets, score orientation or thresholds. The pre-registered E1–E3 design preceded validation scoring; subsequent additions and corrections are labeled explicitly.
- **Saved evidence:** prediction records retain run/config/dataset/encoder identifiers, image/partition IDs, generator, method, score, threshold and predicted label. Tables and plots are derived from stored predictions. Config SHA256 is computed from the four YAML files in sorted-path order.
- **Claim limits:** Results concern this Tiny-GenImage derivative, these generators and frozen *global CLS-token* features. SD15 training is not a replication of SD14-based GenImage settings. Dataset storage shortcuts, content/field-of-view changes, and incomplete control of compression history limit interpretation. No causal artifact localization, universal anomaly-detection ranking, or cross-dataset/deployment guarantee is claimed.

## 7. Supplementary experiment S-BF: content-matched, lossless data

Added after the main study to test whether its conclusions depend on the JPEG/PNG storage split of the Tiny-GenImage derivative. The protocol, partitions and analysis were fixed before any DINOv2 feature of these data was computed, and the final run was executed once.

**Data.** The extended Synthbuster set released with [B-Free](https://github.com/grip-unina/B-Free) (Guillaro et al., CVPR 2025): 1,000 RAISE-1k photographs, resized by the providers to about 1024² pixels at the camera aspect ratio (mostly 1258 × 833 or 1256 × 835, in either orientation), and for each photograph one FLUX and one Stable Diffusion 3.5 image (folder `sd3_large`, 1024 × 1024) generated from its caption. The shared file stem is the **source id**. All files are 8-bit RGB PNG without text or EXIF chunks.

**Audit and gate.** Per file: SHA256, PNG header and chunks, ICC profile, decoded shape, exact RGB hash and dHash, and an 8 × 8 DCT quantization-lattice statistic `L` of luma at all 64 grid offsets (central 256 × 256 region, steps 2 to 10). An image counts as JPEG-encoded at its stored size if `L` on the aligned grid exceeds the median over offsets by more than 0.1. This aligned-grid criterion replaced a first one (maximum over offsets) after the byte audit and before feature extraction, because the search over offsets turned noise into outliers. The gate requires identical id sets, decodable 8-bit RGB PNG files, no flagged stored image while at least 90% of Q96/Q90 re-encoded controls are flagged, and at least 600 eligible ids. **Outcome:** passed; 0 of 3,000 stored images flagged; controls flagged 94.7% (Q96) and 92.7% (Q90); all 1,000 ids eligible.

**Colour.** Real images carry Nikon sRGB (743), Nikon Adobe RGB (256) or no (1) ICC profile; synthetic images carry none. Each tagged real image is converted to standard sRGB with `ImageCms.profileToProfile` (relative colorimetric, flags 0); the untagged image is treated as sRGB. Without the conversion, Adobe RGB values read as sRGB are desaturated, a difference present only in real images. The **sensitivity analysis** repeats fitting, calibration and testing on the 743 sRGB-tagged ids, for which the conversion is checked to be the identity bitwise.

**Partitions.** Ids are linked when any of their images share an exact RGB hash or a dHash within Hamming distance 4; linked ids form one group (952 groups; the largest, 35 ids chained through low-texture images, lies in train). Within two strata (groups whose real images are all sRGB-tagged; other groups), groups sorted by smallest id are permuted with a fresh `default_rng(42)` and cut 60/20/20.

| Partition       | Content                    | Ids (all / sRGB-tagged) | Purpose                                      |
| --------------- | -------------------------- | -----------------------:| -------------------------------------------- |
| `R_train`       | Real, train ids            | 600 / 457               | O1/O2 references, B1 negatives               |
| `F_seen_train`  | SD3.5, train ids           | 600 / 457               | B1 positives only                            |
| `R_cal`         | Real, calibration ids      | 200 / 143               | Thresholds (k = 191; 137 for sRGB-tagged)    |
| `R_test`        | Real, test ids             | 200 / 143               | Real FPR and AUROC negatives                 |
| `F_unseen_test` | FLUX, test ids             | 200 / 143               | Primary positives, never used before testing |
| `F_seen_test`   | SD3.5, test ids            | 200 / 143               | Seen-generator positives                     |

FLUX images of train and calibration ids and SD3.5 images of calibration ids are not used. A development pilot used train ids only, with SD3.5 as positives and no FLUX image.

**Views, encoder and methods.** Primary cell: ViT-S/14 on the 224 native crop (the D-native view; no harmonization because both classes are lossless). Secondary cells: resize to 256 then centre crop (the D-raw view), and ViT-B/14 on the native crop. Encoder, B1, O1, O2, M0 and the calibration rule are those of Sections 3 and 4; B1's cross-validation groups are the id groups, so a real image and its SD3.5 counterpart always share a fold.

**Metrics and statistics.** AUROC; TPR at a matched 5% real FPR, with the threshold at the (m+1)-th largest real test score, m = floor(0.05 n), equal for all methods (an oracle operating point for comparison); TPR and real FPR at the calibrated threshold; matched-pair rate, the share of ids whose synthetic image scores above its own real image. 2,000 bootstrap draws (seed 2026) resample test id groups, with the real, FLUX and SD3.5 images of an id drawn together; all methods and cells share the draws, so differences are paired. A difference is called resolved only if its 95% interval excludes 0. The comparison with the main study is descriptive.

**Claim limits.** The real photographs carry a resampling history and a 3:2 aspect ratio, while the synthetic images are native 1024 × 1024 output. The content match is caption-level, not pixel-level. The colour conversion touches real images only (the sensitivity analysis has none). SD3.5 and FLUX are related latent generators, and there is one unseen generator with 200 test pairs.
