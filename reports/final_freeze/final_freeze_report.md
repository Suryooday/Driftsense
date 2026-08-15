# final Wafer Matching System Freeze and Reproducibility Audit Report

This report presents the results of the final packaging freeze and reproducibility audit for the target registration system.

---

## 1. final System Matching Architecture
The frozen matching pipeline is:
**"Classical NCC-based wafer matching with high-resolution pose refinement"**

- **Stage 1: candidate Generation (coarse)**
  - Normalized Cross-Correlation (NCC) coarse search over translation, coarse scale, and coarse rotation grids.
  - Non-Maximum Suppression (NMS) candidate peak extraction (configured with `nms_radius = 20`).
- **Stage 2: candidate Selection**
  - top-1 candidate selection strictly using classical correlation score (DL verification is bypassed).
- **Stage 3: Sub-pixel Interpolation**
  - local sub-pixel parabolic 1D interpolation in X and Y to refine coordinate centers.
- **Stage 4: High-Resolution Pose Refinement (Coordinate Descent)**
  - *Iteration 1 (Rotation)*: Search rotation span of $\pm 1.0°$ (15 steps) centered at candidate rotation.
  - *Iteration 2 (Scale)*: Search scale drift span of $\pm 0.015$ (11 steps) centered at candidate scale.
  - *Iteration 3 (Fine Rotation)*: Local fine rotation grid search of $\pm 0.2°$ (5 steps) around new best rotation.
  - All evaluations are computed using normalized cross-correlation (NCC) on extracted patches.

---

## 2. Configuration File Location
The immutable configuration parameters are recorded in:
`configs/final_system_config.json`

---

## 3. Deep Learning Exclusion Confirmation
We confirm that **deep learning models (DL V1 and DL V2) are completely excluded** from final matching inference.
- [src/final_system.py](file:///Users/suryodaypratapsingh/Desktop/Semicon/src/final_system.py) does not import `torch` or `torchvision`, and does not load any model checkpoints.
- The imports were verified programmatically by checking `sys.modules` during the reproducibility audit.

---

## 4. Benchmark Integrity Check
The frozen benchmark files under `data/sample_000` through `data/sample_039` have been secured and validated:
- SHA-256 hashes of all input files (reference images, search images, and ground truth json files) have been computed and recorded in:
  `reports/final_freeze/benchmark_hashes.json`
- Subsequent audits will verify benchmark integrity against this manifest.

---

## 5. Reproducibility Results

The final matching pipeline was re-executed on all 40 frozen benchmark samples. The newly reproduced metrics are compared side-by-side with the target baseline results below:

| Metric | Target Baseline | Reproduced Results | Status |
|--------|-----------------|--------------------|--------|
| **Success Rate (%)** | **97.5% (39/40)** | **97.5% (39/40)** | **PASS** |
| **Mean Location Error (px)** | **0.5425 px** | **0.5425 px** | **PASS** |
| **Median Location Error (px)** | **0.5280 px** | **0.5280 px** | **PASS** |
| **Mean Rotation Error (°)** | **0.0910°** | **0.0910°** | **PASS** |
| **Mean Scale Error** | **0.00367** | **0.00367** | **PASS** |

The results are identical to **5 decimal places**, confirming 100% exact numerical reproducibility of the frozen baseline matcher.

---

## 6. List of Files Constituting the Frozen final System

The following files constitute the final matching and validation system:

### Matcher and Configuration
- [configs/final_system_config.json](file:///Users/suryodaypratapsingh/Desktop/Semicon/configs/final_system_config.json): Parameter settings.
- [src/final_system.py](file:///Users/suryodaypratapsingh/Desktop/Semicon/src/final_system.py): Production system entry point.
- [src/hybrid/candidate_generator.py](file:///Users/suryodaypratapsingh/Desktop/Semicon/src/hybrid/candidate_generator.py): Classical candidate generator.
- [src/hybrid/patch_extractor.py](file:///Users/suryodaypratapsingh/Desktop/Semicon/src/hybrid/patch_extractor.py): Image patch warper and extractor.
- [src/matching/classical_matcher.py](file:///Users/suryodaypratapsingh/Desktop/Semicon/src/matching/classical_matcher.py): Core template matching and parabolic subpixel interpolation.

### Audit and Validation Scripts
- [src/audit/final_reproducibility_audit.py](file:///Users/suryodaypratapsingh/Desktop/Semicon/src/audit/final_reproducibility_audit.py): Checks code integrity, imports, and file hashes.
- [src/audit/run_benchmark_validation.py](file:///Users/suryodaypratapsingh/Desktop/Semicon/src/audit/run_benchmark_validation.py): Executes validation matching and reports metrics.
- [reports/final_freeze/benchmark_hashes.json](file:///Users/suryodaypratapsingh/Desktop/Semicon/reports/final_freeze/benchmark_hashes.json): SHA-256 integrity manifest.
- [reports/final_freeze/benchmark_metrics.json](file:///Users/suryodaypratapsingh/Desktop/Semicon/reports/final_freeze/benchmark_metrics.json): Validation evaluation metrics.
- [data/final_predictions_benchmark.json](file:///Users/suryodaypratapsingh/Desktop/Semicon/data/final_predictions_benchmark.json): Matcher predictions on the 40 benchmark samples.
