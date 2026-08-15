"""
Drift Sense - Evaluation Utilities.

Computes matching accuracy metrics such as pixel location error, rotation
error, scale error, and overall match success flags based on ground-truth thresholds.
"""

import math
from typing import Dict, Any


def evaluate_match(
    predicted_x: float,
    predicted_y: float,
    predicted_rot: float,
    predicted_scale: float,
    gt: Dict[str, Any],
    pixel_threshold: float = 3.0,
    rot_threshold: float = 0.5,
    scale_threshold: float = 0.02
) -> Dict[str, Any]:
    """
    Evaluates a predicted match against the ground-truth metadata.

    Args:
        predicted_x: Predicted x-coordinate of the reference center in search image.
        predicted_y: Predicted y-coordinate of the reference center in search image.
        predicted_rot: Predicted rotation of the reference in degrees.
        predicted_scale: Predicted scale factor of the reference relative to search.
        gt: Ground-truth dictionary containing keys: 'true_x', 'true_y', 'rotation_deg', 'scale_factor'.
        pixel_threshold: Max allowed location error (in pixels) for a successful match.
        rot_threshold: Max allowed rotation error (in degrees) for a successful match.
        scale_threshold: Max allowed scale error (absolute) for a successful match.

    Returns:
        A dictionary containing computed errors and success flags:
        {
            'pixel_error': float,
            'rotation_error': float,
            'scale_error': float,
            'success': bool
        }
    """
    # 1. Location Error (Euclidean distance)
    gt_x = gt["true_x"]
    gt_y = gt["true_y"]
    pixel_error = math.sqrt((predicted_x - gt_x) ** 2 + (predicted_y - gt_y) ** 2)

    # 2. Rotation Error (handles wrapping in -180 to 180 or general angles)
    gt_rot = gt["rotation_deg"]
    diff_rot = abs(predicted_rot - gt_rot) % 360.0
    rotation_error = diff_rot if diff_rot <= 180.0 else 360.0 - diff_rot

    # 3. Scale Error (absolute difference on the drift scale factor)
    zoom_ratio = gt.get("zoom_ratio", 10.0)
    pred_drift_scale = predicted_scale / zoom_ratio
    gt_drift_scale = gt.get("drift_scale", gt["scale_factor"] / zoom_ratio)
    scale_error = abs(pred_drift_scale - gt_drift_scale)

    # Determine if matching is successful
    loc_ok = pixel_error <= pixel_threshold
    rot_ok = rotation_error <= rot_threshold
    scale_ok = scale_error <= scale_threshold
    success = loc_ok and rot_ok and scale_ok

    return {
        "pixel_error": pixel_error,
        "rotation_error": rotation_error,
        "scale_error": scale_error,
        "success": success,
        "details": {
            "location_success": bool(loc_ok),
            "rotation_success": bool(rot_ok),
            "scale_success": bool(scale_ok)
        }
    }
