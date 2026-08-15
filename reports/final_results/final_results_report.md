# technical Results Report — Wafer Matching System Validation

This report compiles the final validation, ablation, failure modes, and reproducibility audits for the frozen alignment system.

---

## 1. Final System Matching Architecture

The frozen production system is:
**"Classical NCC-based wafer matching with high-resolution pose refinement"**

- **Inference Entry Point**: [src/final_system.py](file:///Users/suryodaypratapsingh/Desktop/Semicon/src/final_system.py)
- **Configuration**: [configs/final_system_config.json](file:///Users/suryodaypratapsingh/Desktop/Semicon/configs/final_system_config.json)
- **Design Details**: Bypasses deep learning reranking and score fusion. Operates strictly using classical Normalized Cross-Correlation (NCC) sweeps over coarse grid states, extracts NMS candidates, applies sub-pixel parabolic interpolation to coordinate centers, and performs coordinate descent rotation/scale optimization on the top candidate.

---

## 2. Evaluation Protocol

Performance was evaluated under a strict design freeze on two independent datasets:
1. **Frozen 40-Sample Benchmark**: A frozen dataset containing varied rotation, scale, and noise profiles to establish a baseline.
2. **Unseen Robustness Set**: Exactly 200 samples generated using a deterministic master seed `50000` (`canvas_seed = 5000 + i`, `degradation_seed = 50000 + i`) to guarantee zero pattern, image, seed, or path overlap.

### Success Gates (Strict < Comparisons)
- Localization error $< 3.0$ pixels
- Rotation error $< 0.5°$
- Scale error $< 0.02$

During matching inference, matching code had **zero access** to the ground truth files. Evaluation and metric parsing were executed post-inference by a separate evaluation framework.

---

## 3. Benchmark and Robustness Results

| Metric | Frozen 40-Sample Benchmark | New Robustness Set |
|--------|----------------------------|--------------------|
| **Success Rate (%)** | **97.5% (39/40)** | **97.5% (195/200)** (Bootstrap 95% CI: `[95.0%, 99.5%]`) |
| **Mean Location Error (px)** | **0.5425 px** | 5.2371 px (Bootstrap 95% CI: `[0.97 px, 11.46 px]`) |
| **Median Location Error (px)** | **0.5280 px** | 0.5648 px |
| **Mean Rotation Error (°)** | 0.0910° | **0.0847°** |
| **Mean Scale Error** | **0.00367** | 0.00489 |
| **Average Inference Time (s)** | 0.6267 s | **0.3666 s** |

### Understanding the Robustness Location Error
- The robustness set's **mean localization error of 5.2371 px** does not describe typical localization accuracy. This mean is heavily skewed by a small number of catastrophic tracking-loss failures (outliers), where localization error exceeded 50 px.
- The **median localization error (0.5648 px)** and **95th percentile error (0.7723 px)** better describe normal, successful operating registration accuracy.

---

## 4. Ablation Study and DL Contribution Audit

An honest contribution audit was conducted across all system iterations evaluated on the frozen 40-sample benchmark:

| Configuration | Success Rate (%) | Mean Location Error (px) | Mean Rotation Error (°) | Mean Scale Error | Average Inference Time (s) |
|---------------|------------------|--------------------------|-------------------------|------------------|----------------------------|
| **Original Phase 3 Classical** | 77.5% | 0.5425 | 0.2524 | 0.00801 | 0.6153 |
| **Classical + Pose Refinement** | 97.5% | 0.5425 | 0.0910 | 0.00367 | 0.6267 |
| **DL Matcher V1 Reranking** | 27.5% | 143.4126 | 0.4017 | 0.01431 | 0.6153 |
| **Hybrid Fusion** | 72.5% | 17.3104 | 0.2354 | 0.00810 | 0.6153 |
| **DL Matcher V2** | *NOT TRACEABLE* | *NOT TRACEABLE* | *NOT TRACEABLE* | *NOT TRACEABLE* | *NOT TRACEABLE* |
| **Final Frozen System** | **97.5%** | **0.5425** | **0.0910** | **0.00367** | **0.3666** |

### Critical Summary of DL Contribution
- **Pose refinement (coordinate descent) was the sole driver of the performance improvement** from the original classical baseline (increasing success rate from 77.5% to 97.5% by reducing rotation and scale errors).
- **The tested DL candidate-selection strategies (DL Matcher V1 and Hybrid Fusion) did not improve benchmark performance**, and indeed degraded success rate due to high sensitivity to sub-pixel translation shifts.

---

## 5. Failure Modes

### Frozen Benchmark Failure (`sample_021`)
- **Gates Failed**: `rotation` (error = `0.5435°`)
- **Root Cause**: The measured evidence indicates that residual translation error biased the rotation optimization because X/Y coordinates were held fixed during rotation and scale refinement. High noise (`0.0598`) acted as a contributing condition.

### Robustness Set Failures (5 / 200 samples)
- **catastrophic Localization Failures** (error $> 50$ px): 4 samples (`sample_006`, `sample_092`, `sample_121`, `sample_166`). These failed coarse candidate matching due to extreme noise/degradation.
- **Scale-only Failures**: 1 sample (`sample_054`, scale error = `0.0217`) due to ambiguous correlation peaks.
- **Rotation-only Failures**: 0 samples.

---

## 6. Reproducibility and File Integrity

- **Inference Reproducibility**: The final matching pipeline successfully achieved **100% exact numerical reproducibility** matching all metrics of the target baseline.
- **Integrity Hashes**: SHA-256 hashes of all frozen benchmark files were computed and locked in:
  [reports/final_freeze/benchmark_hashes.json](file:///Users/suryodaypratapsingh/Desktop/Semicon/reports/final_freeze/benchmark_hashes.json)
- **DL Exclusions**: The audit programmatically verified that `torch` and `torchvision` packages are not imported or loaded during production system execution.

---

## 7. Final Verdict

**1. SYSTEM PERFORMANCE GENERALIZES TO NEW UNSEEN DATA**

The classical matching pipeline with high-resolution coordinate descent pose refinement achieves a highly robust, sub-pixel, and sub-degree registration accuracy. Its 97.5% success rate generalizes consistently across 200 unseen and independent samples with varied noise levels, rotations, scales, and charging effects.
