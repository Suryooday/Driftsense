"""
DriftSense end-to-end CLI entry point.
Localizes wafer target and computes stage recovery coordinate offsets.
"""
import os
import argparse
import json
import time
import cv2
from src.final_system import FinalSystemMatcher, load_final_config
from src.drift_recovery import DriftRecoveryModule

def main():
    parser = argparse.ArgumentParser(description="Run DriftSense navigation drift detection.")
    parser.add_argument("--reference", required=True, help="Path to reference image (PNG)")
    parser.add_argument("--search", required=True, help="Path to search image (PNG)")
    parser.add_argument("--expected-x", type=float, required=True, help="Expected target X coordinate")
    parser.add_argument("--expected-y", type=float, required=True, help="Expected target Y coordinate")
    parser.add_argument("--output", required=True, help="Path to save output result (JSON)")
    parser.add_argument("--config", default="configs/final_system_config.json", help="Path to system config")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.reference):
        print(f"Error: Reference image not found at {args.reference}")
        return
    if not os.path.exists(args.search):
        print(f"Error: Search image not found at {args.search}")
        return
        
    config = load_final_config(args.config)
    ref_img = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
    srch_img = cv2.imread(args.search, cv2.IMREAD_GRAYSCALE)
    
    # 1. Run final wafer matcher
    matcher = FinalSystemMatcher(args.config)
    pred_res = matcher.match(ref_img, srch_img)
    
    # 2. Run drift recovery calculation
    thresholds = config.get("drift_thresholds", {"aligned_max_px": 1.0, "minor_drift_max_px": 5.0})
    recovery = DriftRecoveryModule(
        aligned_max_px=thresholds["aligned_max_px"],
        minor_drift_max_px=thresholds["minor_drift_max_px"]
    )
    
    expected_target = {"x": args.expected_x, "y": args.expected_y}
    detected_target = {"x": pred_res["predicted_x"], "y": pred_res["predicted_y"]}
    
    drift_res = recovery.calculate_drift(expected_target, detected_target)
    
    # 3. Format output JSON
    output_data = {
        "system": "DriftSense",
        "expected_target": expected_target,
        "detected_target": detected_target,
        "pose": {
            "rotation_degrees": pred_res["predicted_rotation"],
            "scale": pred_res["predicted_scale"]
        },
        "navigation_drift": {
            "dx_pixels": drift_res["dx_pixels"],
            "dy_pixels": drift_res["dy_pixels"],
            "magnitude_pixels": drift_res["magnitude_pixels"]
        },
        "status": drift_res["status"],
        "recommended_correction": drift_res["recommended_correction"],
        "inference_time_seconds": pred_res["elapsed_s"]
    }
    
    # Create output directory
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"DriftSense analysis successfully saved to {args.output}")

if __name__ == "__main__":
    main()
