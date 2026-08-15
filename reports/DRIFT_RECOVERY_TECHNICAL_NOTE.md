# Drift Detection and Stage Coordinate Recovery Technical Note

## 1. Problem Context
During semiconductor wafer inspection, tools repeatedly visit specific site coordinates. Mechanical vibrations, thermal drifts, mechanical slack (backlash), and positioning tolerances can lead to stage navigation drifts. Since semiconductor wafers contain highly repetitive Manhattan-style layouts, drifting off-target can capture a wrong wafer die that still looks visually similar to the expected template. To prevent inspection failures, we must compare the actual detected coordinate of the intended pattern against expected inspection coordinates to recover the correct stage alignment.

---

## 2. Navigation Drift Definition
Navigation drift is the translation vector from the actual target position to the expected coordinate. By tracking this offset, the inspection tool can verify if drift exceeds acceptable tolerances and compute the corrective stage adjustments required to realign the wafer.

---

## 3. Expected vs Detected Coordinates
- **Expected Inspection Coordinates**: The nominal or expected wafer coordinate $(x_{expected}, y_{expected})$ where the inspection tool was commanded to visit.
- **Detected Coordinates**: The localized coordinates $(x_{detected}, y_{detected})$ of the wafer pattern template predicted by the frozen classical matcher.

---

## 4. Mathematical Formulation
The navigation drift components and magnitude are computed as follows:
- **X-axis offset**:
  $$\Delta x = x_{expected} - x_{detected}$$
- **Y-axis offset**:
  $$\Delta y = y_{expected} - y_{detected}$$
- **Drift Magnitude**:
  $$D = \sqrt{\Delta x^2 + \Delta y^2}$$

---

## 5. Coordinate Sign Convention
- **$\Delta x > 0$**: The detected pattern center is to the left of the expected inspection coordinate in the search image. The stage must move in the positive X direction to align.
- **$\Delta y > 0$**: The detected pattern center is shifted vertically relative to the expected coordinate. The stage must apply a positive Y correction vector to align.

---

## 6. Drift Classification
The system uses configuration-driven thresholds (`aligned_max_px = 1.0` and `minor_drift_max_px = 5.0`) to classify the drift status:
- **`ALIGNED`**: $D \le 1.0$ px. The target is aligned; no stage correction is required (`correction_required = False`).
- **`MINOR_DRIFT`**: $1.0 < D \le 5.0$ px. Mild stage drift detected; correction is recommended to maintain centering (`correction_required = True`).
- **`SIGNIFICANT_DRIFT`**: $D > 5.0$ px. Significant stage drift detected; coordinate correction is required (`correction_required = True`).

---

## 7. Recommended Correction
To realign the tool over the correct wafer inspection site, the stage must offset by:
$$\text{Correction}_x = \Delta x$$
$$\text{Correction}_y = \Delta y$$

---

## 8. Demonstration Results
Three synthetic drift scenarios were generated and visualized using sample data:
1. **Aligned Case (`aligned_example.png`)**:
   - Expected: $(447.10, 166.50)$, Detected: $(446.79, 166.34)$
   - Drift: $\Delta x = +0.31$ px, $\Delta y = +0.16$ px, Magnitude = $0.349$ px
   - Status: **`ALIGNED`**
2. **Minor Drift Case (`minor_drift_example.png`)**:
   - Expected: $(450.00, 168.00)$, Detected: $(446.79, 166.34)$
   - Drift: $\Delta x = +3.21$ px, $\Delta y = +1.66$ px, Magnitude = $3.614$ px
   - Status: **`MINOR_DRIFT`**
3. **Significant Drift Case (`significant_drift_example.png`)**:
   - Expected: $(465.00, 175.00)$, Detected: $(446.79, 166.34)$
   - Drift: $\Delta x = +18.21$ px, $\Delta y = +8.66$ px, Magnitude = $20.161$ px
   - Status: **`SIGNIFICANT_DRIFT`**

These cases demonstrate the math verification holds across all status transitions.

---

## 9. Limitations
- **Pixel-Space Limits**: The drift correction vectors are calculated strictly in pixel coordinates. Physical stage displacement (in microns) requires external pixel-to-stage calibration.
- **Inference Reliability**: A template matching failure (e.g. tracking loss due to extreme SEM noise or repeating patterns) will produce incorrect drift values. Stage recovery depends entirely on the accuracy of the underlying registration system.
