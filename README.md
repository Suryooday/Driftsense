# Wafer Pattern Matching and Pose Estimation System

## 1. Overview

Wafer pattern alignment is a critical step in semiconductor fabrication. Given a high-resolution 256x256 wafer reference pattern (template) and a larger 512x512 search image containing environmental degradation (such as SEM noise, charging effects, spatial distortion, and scale changes), the system must determine the sub-pixel coordinates (X, Y), rotation, and scale of the target wafer pattern.

The final production system is:
**"Classical NCC-Based Wafer Pattern Matching with High-Resolution Pose Refinement"**

Deep learning Siamese models (DL V1 and DL V2) were investigated experimentally for candidate reranking. However, they were excluded from the final production inference system because ablation studies proved they did not improve localization performance and introduced sensitivity to sub-pixel translation shifts.

---

## 2. Final System Architecture

```
       Reference Image
             |
             v
      Patch Extraction
             |
             v
Multi-Scale / Rotation Classical NCC Matching
             |
             v
   Top Candidate Selection
             |
             v
    Subpixel Localization
             |
             v
High-Resolution Pose Refinement
  +--------------------------+
  | - Rotation Optimization  |
  | - Scale Optimization     |
  +--------------------------+
             |
             v
  Final X, Y, Rotation, Scale
```

---

## 3. Key Features

- **Sub-Pixel Localization**: Employs 1D parabolic sub-pixel peak interpolation for translation correction.
- **Pose Refinement**: Iterative coordinate descent search over rotation and scale grids using local Normalized Cross-Correlation (NCC) sweeps.
- **Deep Learning Exclusion**: Zero neural network dependency during final inference, eliminating PyTorch overhead and hardware latency.
- **Reproducible Hashes**: SHA-256 integrity manifest securing the 40-sample benchmark set.
- **Robustness Tested**: Validated on 200 independent samples under extreme noise, charging, and scale drifts.
- **Traceability**: All output numbers are mapped to source files via traceability logs.

---

## 4. Results

### Performance Summary

| Dataset | Success Rate (%) | Median Loc Error (px) | Mean Rot Error (°) | Mean Scale Error | Avg Time (s) |
|---|---|---|---|---|---|
| **Frozen 40-Sample Benchmark** | **97.5% (39/40)** | 0.5280 px | 0.0910° | 0.00367 | 0.3666 s |
| **Robustness Set (200 Samples)** | **97.5% (195/200)** | 0.5648 px | 0.0847° | 0.00489 | 0.3666 s |

*Note: The robustness set's mean location error of 5.2371 px is heavily skewed by 4 tracking loss outliers. The median (0.5648 px) and 95th percentile (0.7723 px) represent typical operating accuracy.*

---

## 5. Experimental Ablation

| Configuration | Success Rate (%) | Mean Loc Error (px) | Mean Rot Error (°) | Mean Scale Error | Avg Time (s) |
|---|---|---|---|---|---|
| Original Phase 3 Classical | 77.5% | 0.5425 | 0.2524° | 0.00801 | 0.6153 s |
| Classical + Pose Refinement | 97.5% | 0.5425 | 0.0910° | 0.00367 | 0.6267 s |
| DL Matcher V1 Reranking | 27.5% | 143.4126 | 0.4017° | 0.01431 | 0.6153 s |
| Hybrid Fusion | 72.5% | 17.3104 | 0.2354° | 0.00810 | 0.6153 s |
| **Final Frozen System** | **97.5%** | **0.5425** | **0.0910°** | **0.00367** | **0.3666 s** |

*Scientific Interpretation*: Bypassing deep learning and applying classical coordinate descent refinement yielded the success rate increase (from 77.5% to 97.5%). The DL models did not improve candidate selection on the benchmark due to translation-rotation coupling.

---

## 6. Installation

Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 7. Running Inference

Evaluate a single reference and search image pair:
```bash
python3 -m src.run_final \
    --reference data/sample_000/reference_image.png \
    --search data/sample_000/search_image.png \
    --output results/prediction.json
```

---

## 8. Running Demo

Run the matching demo on success and failure wafer samples:
```bash
python3 -m src.demo
```

---

## 9. Reproducibility

Run the single-command validation check to verify hashes and baseline reproduction:
```bash
python3 -m src.verify_final_system
```

---

## 10. Project Structure

```
.
├── README.md
├── requirements.txt
├── config.yaml
├── configs/
│   └── final_system_config.json
├── data/
│   ├── sample_000/ to sample_039/ (Frozen Benchmark)
│   └── robustness_samples/ (200 Independent Samples)
├── reports/
│   ├── final_freeze/ (Integrity hashes & benchmark metrics)
│   ├── final_results/ (Plots, reports, & traceability logs)
│   ├── FINAL_TECHNICAL_REPORT.md
│   └── PROJECT_PRESENTATION.md
├── src/
│   ├── final_system.py (Production Entry Point)
│   ├── run_final.py (CLI tool)
│   ├── demo.py (Visual demonstration runner)
│   ├── verify_final_system.py (Reproducibility suite)
│   ├── audit/ (Auditing & plotting helpers)
│   ├── visualization/ (Individual & composite plot script)
│   ├── matching/ (Classical sub-pixel template matcher)
│   └── hybrid/ (Patch extraction & candidate generation)
└── venv/
```

---

## 11. Navigation Drift Detection and Recovery

The DriftSense coordinate recovery layer compares the expected nominal inspection coordinates against the actual detected targets. It computes coordinate corrections:
- $\Delta x = x_{expected} - x_{detected}$
- $\Delta y = y_{expected} - y_{detected}$
- $Drift\ Magnitude = \sqrt{\Delta x^2 + \Delta y^2}$

### Stage Control Disclaimer
The current implementation provides pixel-space coordinate recovery. Conversion to physical stage displacement units such as microns requires a calibrated pixel-to-stage transformation matrix supplied by the inspection tool's hardware interface. The system does not directly command or actuate physical stages.
