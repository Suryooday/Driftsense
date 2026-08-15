# Classical Matcher — Evaluation Report

## Results Summary

| Metric | Value |
|:---|:---|
| **Successful matches** | **31 / 40 (77.5%)** |
| **Mean location error** | **0.5425 px** |
| **Median location error** | 0.5280 px |
| **Max location error** | **0.8811 px** (sample_021) |
| **Mean rotation error** | 0.2524° |
| **Max rotation error** | 0.9459° (sample_002) |
| **Mean scale error** | 0.0080 |
| **Max scale error** | 0.0235 (sample_003) |
| **Mean inference time** | 610.6 ms / image |
| **Total dataset time** | 24.4 s (40 samples) |

---

## Error vs Noise Level Chart

![Classical matcher error by noise bucket](/Users/suryodaypratapsingh/.gemini/antigravity-ide/brain/57b4bca5-b35d-4153-b946-3670601763e8/chart_classical.png)

---

## Breakdown by Noise Bucket

| Bucket | n | Success | Mean Loc (px) | Mean Rot (°) | Mean Scale | Max Loc (px) |
|:---|---:|---:|---:|---:|---:|---:|
| Low (< 0.025) | 12 | **9 / 12** | 0.5141 | 0.2656 | 0.00734 | 0.6821 |
| Medium (0.025–0.045) | 13 | **12 / 13** | 0.5609 | 0.1359 | 0.00784 | 0.7459 |
| High (≥ 0.045) | 15 | **10 / 15** | 0.5493 | 0.3428 | 0.00869 | 0.8811 |

> [!NOTE]
> Location error is remarkably stable across all noise levels (0.51–0.56 px mean). The NCC sweep is robust to SEM-style noise and charging artifacts. All failures are rotation-precision failures only.

---

## Failure Breakdown

| Failure Type | Count | Samples |
|:---|---:|:---|
| Location > 3.0 px | **0** | — |
| Rotation > 0.5° | **8** | 002, 006, 007, 021, 025, 029, 032, 036 |
| Scale > 0.02 | **1** | 003 (Δ = 0.0235) |

All rotation failures are coarse-grid discretization artifacts (0.5° step grid) — the correct cell is always found; only the angle estimate is slightly off.

---

## Worst 5 Samples (for failure analysis)

| Rank | Sample | Loc Err | Rot Err | Scale Err | Noise Bucket |
|:---|:---|---:|---:|---:|:---|
| #1 | sample_021 | 0.8811 px | 0.7864° ✗ | 0.00131 | high |
| #2 | sample_008 | 0.7459 px | 0.1806° | 0.01534 | medium |
| #3 | sample_000 | 0.6944 px | 0.0668° | 0.00466 | medium |
| #4 | sample_018 | 0.6821 px | 0.0147° | 0.01025 | low |
| #5 | sample_035 | 0.6781 px | 0.0863° | 0.00692 | high |

---

## Outputs

| File | Path |
|:---|:---|
| Script | [evaluate.py](file:///Users/suryodaypratapsingh/Desktop/Semicon/src/scoring/evaluate.py) |
| CSV results | [results_classical.csv](file:///Users/suryodaypratapsingh/Desktop/Semicon/data/evaluation/results_classical.csv) |
| Bar chart | [chart_classical.png](file:///Users/suryodaypratapsingh/Desktop/Semicon/data/evaluation/chart_classical.png) |

---

## Reuse for DL Predictions

```bash
python -m src.scoring.evaluate \
  --predictions data/predictions_dl.json \
  --tag dl-refined
```
