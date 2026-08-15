# Wafer Pattern Matching System — Slide Presentation Outline

This document outlines a 10-slide presentation deck covering the design, results, and auditing of the wafer alignment project.

---

## Slide 1 — Title
- **Title**: High-Accuracy sub-Pixel Wafer Pattern Registration
- **Main Points**:
  - wafer pattern alignment under SEM noise, scale drift, and rotations.
  - Classical NCC matching coupled with high-resolution pose refinement.
  - Bypassing deep learning to maximize reliability and speed.
- **Recommended Figure**: `reports/final_results/figures/success_gallery.png`
- **Speaker Notes**: "Welcome. Today I am presenting our Wafer Alignment System. We built a robust classical registration pipeline that outperforms deep learning in accuracy, speed, and determinism."

---

## Slide 2 — Problem
- **Title**: wafer Alignment Challenges in Semiconductor Fab
- **Main Points**:
  - Environmental noise in SEM imaging (Gaussian, line charging).
  - Spatial scaling drifts and angular rotations.
  - Strict registration constraints: $< 3$ px translation error, $< 0.5°$ rotation error, $< 0.02$ scale error.
- **Recommended Figure**: `reports/final_results/alignment_examples/success/sample_000_alignment.png` (Panel B: Search Image)
- **Speaker Notes**: "SEM wafer imaging introduces severe noise and charging lines. Our goal is to align a 256x256 reference template inside a 512x512 search image with sub-pixel and sub-degree accuracy."

---

## Slide 3 — Input and Output
- **Title**: Inputs, Outputs, and Evaluation Gates
- **Main Points**:
  - *Input*: Grayscale Reference image (256x256), Search image (512x512).
  - *Output*: Center coordinate (X, Y), Rotation (degrees), Scale factor.
  - *Success criteria*: Success requires passing all three error gates simultaneously.
- **Recommended Figure**: `reports/final_results/alignment_examples/success/sample_000_alignment.png` (Panel D: Error Summary)
- **Speaker Notes**: "Our system accepts grayscale images and outputs predicted pose values. Success is evaluated post-inference via three strict error gates."

---

## Slide 4 — System Architecture
- **Title**: final System Architecture
- **Main Points**:
  - Coarse NCC template matching over discrete X/Y/rotation/scale grids.
  - Top-1 candidate selection using classical correlation scores.
  - Sub-pixel parabolic interpolation to refine coordinate centers.
  - Coordinate descent refinement (Rotation sweep -> Scale sweep -> Fine rotation sweep).
- **Recommended Figure**: Diagram in README Section 2 (System Architecture flowchart)
- **Speaker Notes**: "Here is the layout of our pipeline. We use classical NCC matching to locate candidates, then sub-pixel interpolate the coordinates, and finally apply coordinate descent to refine rotation and scale."

---

## Slide 5 — Classical NCC Matching
- **Title**: Coarse Matching & Candidate Peak Selection
- **Main Points**:
  - Normalized Cross-Correlation (NCC) slides template across search space.
  - Coarse rotation discretized to 13 steps; coarse scale to 7 steps.
  - Non-Maximum Suppression (NMS) with a radius of 20 pixels extracts local correlation peaks.
  - Top-1 candidate is selected strictly using classical correlation score.
- **Recommended Figure**: `reports/final_results/alignment_examples/success/sample_010_alignment.png` (Panel A & B)
- **Speaker Notes**: "Coarse template matching identifies the candidate peaks. We run local NCC sweeps and use NMS to isolate the best candidate pattern tile."

---

## Slide 6 — Pose Refinement
- **Title**: High-Resolution Pose Refinement
- **Main Points**:
  - Translation coordinates are refined using parabolic sub-pixel peak interpolation.
  - Coordinate descent optimizes rotation and scale sequentially.
  - Iteration 1: Rotation sweep ($\pm 1.0°$ in 15 steps).
  - Iteration 2: Scale sweep ($\pm 0.015$ in 11 steps).
  - Iteration 3: Fine rotation sweep ($\pm 0.2°$ in 5 steps).
- **Recommended Figure**: `reports/final_results/alignment_examples/success/sample_020_alignment.png` (Panel C: Local Zoom & Boundaries)
- **Speaker Notes**: "Once the coarse peak is found, we refine translation using parabolic interpolation. We then perform coordinate descent sweeps on rotation and scale, achieving sub-degree and sub-pixel registration."

---

## Slide 7 — Deep Learning Experiment and Ablation
- **Title**: Siamese CNN Investigation & Ablation
- **Main Points**:
  - Siamese ResNet-18 model trained to verify candidate tiles.
  - Ablation results showed DL candidate reranking decreased success rate to 27.5%.
  - Translation-rotation coupling: sub-pixel translation shifts corrupted Siamese similarity predictions.
  - DL was completely excluded from final inference.
- **Recommended Figure**: `reports/final_results/figures/figure4_ablation_comparison.png` (Ablation Success Rate Comparison)
- **Speaker Notes**: "We evaluated Siamese deep learning models. Reranking actually degraded success rates because sub-pixel translation offsets corrupted similarity features. Thus, we excluded DL from production."

---

## Slide 8 — Final Results
- **Title**: Validation and Robustness Results
- **Main Points**:
  - *Frozen Benchmark (40 samples)*: 97.5% success rate, median loc error 0.528 px, mean rot error 0.091°.
  - *Robustness Set (200 samples)*: 97.5% success rate, median loc error 0.565 px, mean rot error 0.085°.
  - Average inference time: 0.366 s (faster due to PyTorch exclusion).
- **Recommended Figure**: `reports/final_results/figures/figure1_success_rate.png` (Success Rate Comparison)
- **Speaker Notes**: "Our final system achieved 97.5% success on both the benchmark and the 200-sample robustness set. Processing time dropped to 0.36 seconds per sample because we eliminated PyTorch inference overhead."

---

## Slide 9 — Failure Analysis and Robustness
- **Title**: Failure Modes & Noise Sensitivity
- **Main Points**:
  - Benchmark failure (`sample_021`): fixed-center refinement biased rotation search due to 0.88 px translation offset.
  - Robustness failures: 4 catastrophic tracking losses (SEM noise std $> 0.05$), 1 minor scale failure.
  - Success rates remain high across noise levels (100% in low noise, 96.6% in high noise).
- **Recommended Figure**: `reports/final_results/figures/failure_gallery.png` (Composite Failure Gallery)
- **Speaker Notes**: "Only 5 out of 200 robustness samples failed—4 were catastrophic tracking losses under extreme noise. We also analyzed the benchmark failure, which was caused by translation-rotation coupling."

---

## Slide 10 — Conclusion
- **Title**: Conclusion & Deliverables
- **Main Points**:
  - **Verdict**: System performance generalizes reliably to unseen wafer patterns and environmental degradations.
  - **Packaging**: CLI tool (`run_final.py`), demo script (`demo.py`), and validation suite (`verify_final_system.py`) created.
  - **Reproducibility**: File hashes are locked and validated (100% exact numerical reproduction achieved).
- **Recommended Figure**: `reports/final_results/figures/success_gallery.png`
- **Speaker Notes**: "To conclude, our classical wafer matching system generalizes beautifully. All files are fully packaged, verified, and reproducibility checks pass. Thank you."
