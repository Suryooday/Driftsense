# failure Analysis Summary

## 1. Frozen Benchmark failure

Exactly **1 out of 40 samples** failed in the frozen benchmark:

- **Sample ID**: `sample_021`
- **Localization Error**: `0.8811` px (OK)
- **Rotation Error**: `0.5435°` (FAILED, threshold < 0.5°)
- **Scale Error**: `0.0077` (OK)
- **Failed Success Criterion**: `rotation`
- **Analysis**: Residual translation error of 0.88 px biased the rotation search because X/Y center coordinates were held fixed during coordinate descent refinement.

## 2. Robustness Set failures

Exactly **5 out of 200 samples** (2.50%) failed the success gates:

- **catastrophic Localization failures** (error > 50 px): 4
- **Rotation-only failures**: 0
- **Scale-only failures**: 1

| Sample ID | Location Error (px) | Rotation Error (°) | Scale Error | Failed Criteria | failure Category |
|---|---|---|---|---|---|
| `sample_006` | 119.9243 | 0.1797 | 0.00244 | localization | Extreme noise/degradation |
| `sample_054` | 0.6390 | 0.0229 | 0.02174 | scale | Ambiguous correlation peaks |
| `sample_092` | 81.2483 | 0.2776 | 0.04920 | localization, scale | Extreme noise/degradation |
| `sample_121` | 415.0787 | 0.0547 | 0.00239 | localization | Ambiguous correlation peaks |
| `sample_166` | 320.1853 | 0.5110 | 0.02154 | localization, rotation, scale | Coupling of translation error into rotation refinement |
