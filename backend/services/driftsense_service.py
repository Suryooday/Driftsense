"""
DriftSense service layer.
Wraps the frozen FinalSystemMatcher and DriftRecoveryModule into a single
analysis call that the API endpoints consume.
"""
import numpy as np
import cv2
from typing import Dict, Any

from src.final_system import FinalSystemMatcher, load_final_config
from src.drift_recovery import DriftRecoveryModule


class DriftSenseService:
    """Singleton-style service that holds a warm matcher instance."""

    def __init__(self, config_path: str = "configs/final_system_config.json") -> None:
        self.config = load_final_config(config_path)
        self.matcher = FinalSystemMatcher(config_path)

        thresholds = self.config.get("drift_thresholds", {})
        self.recovery = DriftRecoveryModule(
            aligned_max_px=thresholds.get("aligned_max_px", 1.0),
            minor_drift_max_px=thresholds.get("minor_drift_max_px", 5.0),
        )

    def run_analysis(
        self,
        reference: np.ndarray,
        search: np.ndarray,
        expected_x: float,
        expected_y: float,
    ) -> Dict[str, Any]:
        """
        End-to-end: match → drift → correction.
        Returns a flat dict ready for AnalysisResponse construction.
        """
        # 1. Run frozen matcher
        match_result = self.matcher.match(reference, search)

        pred_x = match_result["predicted_x"]
        pred_y = match_result["predicted_y"]

        # Guard against matching failure
        if pred_x is None or pred_y is None or match_result.get("found", 1) == 0:
            sh, sw = search.shape[:2]
            return {
                "expected": {"x": expected_x, "y": expected_y},
                "detected": {"x": 0.0, "y": 0.0},
                "drift": {"dx": 0.0, "dy": 0.0, "magnitude": 0.0, "status": "MATCH_FAILED"},
                "pose": {"rotation": 0.0, "scale": 0.0},
                "confidence": {"ncc_score": float(match_result.get("confidence_score", 0.0))},
                "stage_correction": {"move_x": 0.0, "move_y": 0.0},
                "inference_time_s": match_result["elapsed_s"],
                "search_width": float(sw),
                "search_height": float(sh),
            }



        # 2. Run drift recovery
        drift_result = self.recovery.calculate_drift(
            expected_target={"x": expected_x, "y": expected_y},
            detected_target={"x": pred_x, "y": pred_y},
        )

        sh, sw = search.shape[:2]

        return {
            "expected": {"x": expected_x, "y": expected_y},
            "detected": {"x": round(pred_x, 4), "y": round(pred_y, 4)},
            "drift": {
                "dx": round(drift_result["dx_pixels"], 4),
                "dy": round(drift_result["dy_pixels"], 4),
                "magnitude": round(drift_result["magnitude_pixels"], 4),
                "status": drift_result["status"],
            },
            "pose": {
                "rotation": round(match_result["predicted_rotation"], 4),
                "scale": round(match_result["predicted_scale"], 4),
            },
            "confidence": {
                "ncc_score": round(match_result["confidence_score"], 4),
            },
            "stage_correction": {
                "move_x": round(drift_result["recommended_correction"]["x_pixels"], 4),
                "move_y": round(drift_result["recommended_correction"]["y_pixels"], 4),
            },
            "inference_time_s": round(match_result["elapsed_s"], 4),
            "search_width": float(sw),
            "search_height": float(sh),
        }

