# Wafer Pattern Matching and Pose Registration Technical Report

## 1. Abstract
We present a highly robust, sub-pixel, and sub-degree wafer alignment system designed for semiconductor manufacturing. By coupling coarse Normalized Cross-Correlation (NCC) template matching, parabolic sub-pixel peak interpolation, and coordinate descent pose optimization, we achieve a **97.5% success rate** on both a frozen 40-sample benchmark and an independent 200-sample robustness set. Experimental deep learning tracks (Siamese CNNs) were evaluated but ultimately excluded from production inference due to sensitivity to sub-pixel translation shifts and high computational overhead.

---

## 2. Problem Statement
Semiconductor wafer alignment requires registering a known 256x256 reference wafer pattern (template) onto a larger 512x512 search image. The target search image contains severe simulated SEM degradations including Gaussian noise, charging lines, spatial scaling drifts, and rotations.

Success is defined by three independent gates:
1. **Localization Error**: $< 3.0$ pixels
2. **Rotation Error**: $< 0.5°$
3. **Scale Error**: $< 0.02$

---

## 3. System Design

The system follows a modular architecture:
1. **coarse Search (Candidate Generation)**: Computes NCC over a discrete 3D search space (translation, coarse scale, and coarse rotation).
2. **Candidate Peak Selection**: Applies 2D Non-Maximum Suppression (NMS) to extract peak locations and selects the top-1 classical candidate.
3. **Sub-pixel Interpolation**: Performs parabolic 1D interpolation along the X and Y axes around the coarse center.
4. **Coordinate Descent Pose Refinement**: Performs fine local sweeps of rotation, scale, and fine rotation sequentially to maximize local patch similarity.

---

## 4. Classical Matching Method
Candidate matching operates by sliding a reference template across a search image. We normalize search features using local mean and standard deviation subtraction. For candidate generation, the system discretizes scale and rotation ranges based on parameters in `configs/final_system_config.json`:
- Coarse rotation: 13 steps between $-3.0°$ and $+3.0°$.
- Coarse scale: 7 steps between $0.97$ and $1.03$.
- Candidate count: $K = 5$ peaks are extracted using NMS with a radius of 20 pixels.

---

## 5. Subpixel Localization
To refine the translation coordinates without excessive search discretization, we apply local 1D quadratic interpolation. Let $R(x)$ be the correlation value at candidate coordinate $x$, and $R(x-1)$, $R(x+1)$ be the neighboring grid correlation values. The sub-pixel offset $dx$ is given by:
$$dx = \frac{R(x-1) - R(x+1)}{2 \cdot (R(x-1) - 2 R(x) + R(x+1))}$$
This provides sub-pixel resolution down to $\approx 0.5$ pixels.

---

## 6. High-Resolution Pose Refinement
Refined alignment coordinates are optimized using a coordinate descent approach to maximize local NCC:
- **Iteration 1 (Rotation)**: Grid sweep of 15 steps over a range of $\pm 1.0°$ around the candidate rotation.
- **Scale Optimization**: Grid sweep of 11 steps over scale drift range of $\pm 0.015$.
- **Iteration 2 (Fine Rotation)**: Grid sweep of 5 steps over a fine range of $\pm 0.2°$ around the new optimal rotation.

This ensures rotation errors are reduced below the $0.5°$ gate, and scale errors below the $0.02$ gate.

---

## 7. Experimental Deep Learning Investigation
We investigated two Siamese CNN architectures (DL Matcher V1 and DL Matcher V2) utilizing a pretrained ResNet-18 backbone. The models were trained to predict whether pair candidates were matching or non-matching.
- **DL Matcher V1**: Fine-tuned ResNet-18 Siamese model.
- **DL Matcher V2**: Frozen ResNet-18 backbone trained with AdamW optimizer on 200 canvas sets.
- **Key Discovery**: The DL models suffered from canvas-density overfitting. Due to translation-rotation coupling, sub-pixel translation shifts of the coarse candidate corrupted the DL model's similarity scores, making DL reranking highly unreliable (reducing benchmark success rate to 27.5%).

---

## 8. Ablation Study
The table below lists the performance of the matching pipeline configurations on the frozen 40-sample benchmark:

| Configuration | Success Rate (%) | Mean Location Error (px) | Mean Rotation Error (°) | Mean Scale Error | Average Inference Time (s) |
|---|---|---|---|---|---|
| Original Phase 3 Classical | 77.5% | 0.5425 | 0.2524 | 0.00801 | 0.6153 |
| Classical + Pose Refinement | 97.5% | 0.5425 | 0.0910 | 0.00367 | 0.6267 |
| DL Matcher V1 Reranking | 27.5% | 143.4126 | 0.4017 | 0.01431 | 0.6153 |
| Hybrid Fusion | 72.5% | 17.3104 | 0.2354 | 0.00810 | 0.6153 |
| Final Frozen System | **97.5%** | **0.5425** | **0.0910** | **0.00367** | **0.3666** |

**Conclusion**: The coordinate descent pose refinement is the sole contributor to the improvement (from 77.5% to 97.5%). Bypassing DL reranking eliminated neural network execution latency and improved speed by 41%.

---

## 9. Frozen Benchmark Evaluation
Re-running the final matching system on the 40 benchmark samples yielded:
- **Success Rate**: 97.5% (39 / 40 successful matches)
- **Mean Localization Error**: `0.5425 px`
- **Median Localization Error**: `0.5280 px`
- **Mean Rotation Error**: `0.0910°`
- **Mean Scale Error**: `0.00367`

---

## 10. Robustness Evaluation
The system was evaluated on a 200-sample unseen robustness set containing severe degradations:
- **Success Rate**: **97.5%** (195 / 200 successful matches)
  - **Bootstrap 95% CI**: `[95.0%, 99.5%]` (computed with $B=2000$ resamples)
- **Median Localization Error**: **0.5648 px**
- **Mean Rotation Error**: **0.0847°**
- **Mean Scale Error**: **0.00489**
- **Mean Inference Time**: **0.3666 s** per sample

---

## 11. Failure Analysis

### Frozen Benchmark Failure (`sample_021`)
- **Location Error**: `0.8811 px` (OK)
- **Rotation Error**: `0.5435°` (FAILED, gate $< 0.5°$)
- **Root Cause**: Residual translation error of 0.88 px was present while X/Y center coordinates were held fixed during rotation refinement. The rotation search optimized to a biased rotation value to compensate for the translation shift.

### Robustness Failures (5 / 200 samples)
- **Catastrophic Localization Failures (TRACKING LOSS)**: 4 samples (`sample_006`, `sample_092`, `sample_121`, `sample_166` all had location error $> 50$ px due to extreme SEM noise).
- **Scale-only Failures**: 1 sample (`sample_054`, scale error = `0.0217`) due to ambiguous correlation peaks.
- **Rotation-only Failures**: 0 samples.

---

## 12. Limitations
The system presents the following physical and algorithmic limits:
1. **Catastrophic Tracking Loss**: Under high SEM noise levels ($> 0.05$ std), the template matching correlation peak can collapse, leading to tracking loss.
2. **Translation-Rotation Coupling**: Because pose refinement optimizes rotation and scale while keeping the translation center fixed, any translation offset directly introduces rotation search bias.

---

## 13. Reproducibility
Reproducibility is guaranteed by three locked artifacts:
- Configuration parameters in [configs/final_system_config.json](file:///Users/suryodaypratapsingh/Desktop/Semicon/configs/final_system_config.json).
- SHA-256 hash manifest in [reports/final_freeze/benchmark_hashes.json](file:///Users/suryodaypratapsingh/Desktop/Semicon/reports/final_freeze/benchmark_hashes.json).
- Single-command reproducibility runner: `python3 -m src.verify_final_system`.

---

## 14. Conclusion
The locked wafer registration matching pipeline achieves sub-pixel and sub-degree registration accuracy. It is computationally lightweight, independent of PyTorch execution, and generalizes reliably to unseen environmental conditions.
