"""
 Drift Recovery Module.
Calculates pixel-space navigation drift, recommends coordinate corrections,
and classifies status based on configurable thresholds.
"""
import numpy as np
from typing import Dict, Any

class DriftRecoveryModule:
    def __init__(self, aligned_max_px: float = 1.0, minor_drift_max_px: float = 5.0) -> None:
        self.aligned_max_px = aligned_max_px
        self.minor_drift_max_px = minor_drift_max_px
        
    def calculate_drift(
        self,
        expected_target: Dict[str, float],
        detected_target: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calculates translation drift from expected coordinates to actual localized target.
        
        Sign Convention:
          dx = expected_x - detected_x
          dy = expected_y - detected_y
          
          dx > 0: Detected target is to the left of expected target (stage must move right, i.e., positive X correction).
          dy > 0: Detected target is above/below expected target according to image coordinate system.
        """
        ex, ey = expected_target["x"], expected_target["y"]
        dx_val, dy_val = detected_target["x"], detected_target["y"]
        
        dx = float(ex - dx_val)
        dy = float(ey - dy_val)
        drift_magnitude = float(np.sqrt(dx**2 + dy**2))
        
        # Classification logic
        if drift_magnitude <= self.aligned_max_px:
            status = "ALIGNED"
            correction_required = False
        elif drift_magnitude <= self.minor_drift_max_px:
            status = "MINOR_DRIFT"
            correction_required = True
        else:
            status = "SIGNIFICANT_DRIFT"
            correction_required = True
            
        return {
            "dx_pixels": dx,
            "dy_pixels": dy,
            "magnitude_pixels": drift_magnitude,
            "status": status,
            "correction_required": correction_required,
            "recommended_correction": {
                "x_pixels": dx,
                "y_pixels": dy
            },
            "calibration_disclaimer": "Physical stage displacement requires external pixel-to-stage calibration."
        }
