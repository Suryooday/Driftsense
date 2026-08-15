# Ablation Study results

| System | Success Rate (%) | Mean Location Error (px) | Mean Rotation Error (°) | Mean Scale Error | Average Inference Time (s) |
|---|---|---|---|---|---|
| Original Phase 3 Classical | 77.5% | 0.5425 | 0.2524 | 0.00801 | 0.6153 |
| Classical + Pose Refinement | 97.5% | 0.5425 | 0.0910 | 0.00367 | 0.6267 |
| DL Matcher V1 Reranking | 27.5% | 143.4126 | 0.4017 | 0.01431 | 0.6153 |
| Hybrid Fusion | 72.5% | 17.3104 | 0.2354 | 0.00810 | 0.6153 |
| DL Matcher V2 | NOT TRACEABLE | NOT TRACEABLE | NOT TRACEABLE | NOT TRACEABLE | NOT TRACEABLE |
| Final Frozen System | 97.5% | 0.5425 | 0.0910 | 0.00367 | 0.3666 |


*Note: Pose refinement (coordinate descent) was the measurable source of improvement from the original classical baseline to the final system, while the tested DL candidate-selection strategies did not improve benchmark performance.*