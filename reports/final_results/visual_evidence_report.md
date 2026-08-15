# Visual Evidence Report — Wafer Matching System Validation

This report presents the visual alignment examples, failure diagnostics, composite galleries, and visualization integrity audits for the frozen matching system.

---

## 1. Successful Examples

We generated four representative successful visualizations:
1. **`sample_000` (Frozen Benchmark)**: Representative of low-noise alignment. The predicted center matches the ground truth center identically, and the aligned reference boundary overlaps perfectly on the search template.
2. **`sample_010` (Robustness Set)**: Representative of low-noise robustness matching.
3. **`sample_020` (Robustness Set)**: Representative of medium-noise robustness matching.
4. **`sample_030` (Robustness Set)**: Representative of high-noise robustness matching.

These samples demonstrate that the sub-pixel template matcher successfully locks onto patterns and generalizes to varied noise distributions, scale drifts, and rotations.

The individual success plots are located at:
[reports/final_results/alignment_examples/success/](file:///Users/suryodaypratapsingh/Desktop/Semicon/reports/final_results/alignment_examples/success/)

The summary gallery of success cases is saved at:
![Success Gallery](file:///Users/suryodaypratapsingh/Desktop/Semicon/reports/final_results/figures/success_gallery.png)

---

## 2. Frozen Benchmark Failure Analysis (`sample_021`)

- **Failed Gate**: `rotation` (error = `0.5435°` vs Success Threshold `< 0.5°`)
- **Visual Evidence**:
  - The local zoom panel highlights a translation center offset of `0.8811` pixels.
  - The transformation boundary overlay demonstrates that this translation mismatch directly biased the subsequent rotation refinement.
- **Root Cause**:
  > **"Residual translation error was present while X/Y coordinates were held fixed during rotation and scale refinement. The measured NCC objective selected a biased rotation under this fixed-center condition."**
- **Refinement Trace**:
  - *Intermediate refinement trace unavailable from saved outputs.*

The detailed failure panel plot is located at:
[reports/final_results/alignment_examples/frozen_failure/sample_021/sample_021_failure.png](file:///Users/suryodaypratapsingh/Desktop/Semicon/reports/final_results/alignment_examples/frozen_failure/sample_021/sample_021_failure.png)

---

## 3. Robustness Failures (5 / 200 samples)

Exactly 5 samples failed the success criteria. The individual visualizations are located under [reports/final_results/alignment_examples/robustness_failures/](file:///Users/suryodaypratapsingh/Desktop/Semicon/reports/final_results/alignment_examples/robustness_failures/).

1. **`sample_006`**:
   - **Failed Gate**: `localization` (error = `119.92` px)
   - **Status**: **TRACKING LOSS**
   - **Category**: Extreme noise/degradation
2. **`sample_054`**:
   - **Failed Gate**: `scale` (error = `0.0217`)
   - **Status**: Minor scale mismatch
   - **Category**: Ambiguous correlation peaks
3. **`sample_092`**:
   - **Failed Gates**: `localization` (error = `81.25` px), `scale` (error = `0.0492`)
   - **Status**: **TRACKING LOSS**
   - **Category**: Extreme noise/degradation
4. **`sample_121`**:
   - **Failed Gate**: `localization` (error = `415.08` px)
   - **Status**: **TRACKING LOSS**
   - **Category**: Ambiguous correlation peaks
5. **`sample_166`**:
   - **Failed Gates**: `localization` (error = `320.19` px), `rotation` (error = `0.5110°`), `scale` (error = `0.0215`)
   - **Status**: **TRACKING LOSS**
   - **Category**: Coupling of translation error into rotation refinement

The summary composite gallery of all failure cases is saved at:
![Failure Gallery](file:///Users/suryodaypratapsingh/Desktop/Semicon/reports/final_results/figures/failure_gallery.png)

---

## 4. Visualization Integrity Audit

We verify that the visual assets conform to strict validation guidelines:
- **Prediction loading**: Loaded strictly from existing saved predictions (`final_predictions_benchmark.json` and `predictions.json`).
- **Ground Truth Isolation**: Ground truth files were used solely for overlay rendering (markers/boundaries) and error calculations in the final plotting steps. Ground truth was **not** accessed or used by any matching script.
- **System Integrity**: No matcher parameters, algorithm files, prediction files, or input dataset splits were modified.
- **Traceability**: All visual assets are fully trace-mapped in [visualization_traceability.json](file:///Users/suryodaypratapsingh/Desktop/Semicon/reports/final_results/visualization_traceability.json).
