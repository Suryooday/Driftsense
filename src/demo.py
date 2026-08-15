"""
Demo script executing the frozen matching system on representative samples and reporting errors.
"""
import os
import json
import numpy as np
import cv2
from src.final_system import FinalSystemMatcher

def run_sample_demo(matcher: FinalSystemMatcher, sample_id: str, sample_dir: str):
    ref_path = os.path.join(sample_dir, "reference_image.png")
    srch_path = os.path.join(sample_dir, "search_image.png")
    gt_path = os.path.join(sample_dir, "ground_truth.json")
    
    if not os.path.exists(ref_path) or not os.path.exists(srch_path):
        print(f"Error: Images for {sample_id} not found.")
        return
        
    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    srch = cv2.imread(srch_path, cv2.IMREAD_GRAYSCALE)
    
    # Run matcher
    pred = matcher.match(ref, srch)
    
    # Compute error metrics if ground truth is available
    has_gt = os.path.exists(gt_path)
    loc_err = None
    rot_err = None
    scale_err = None
    status = "SUCCESS"
    
    if has_gt:
        with open(gt_path, "r") as f:
            gt = json.load(f)
            
        loc_err = float(np.sqrt((pred["predicted_x"] - gt["true_x"])**2 + (pred["predicted_y"] - gt["true_y"])**2))
        
        raw_rot_err = abs(pred["predicted_rotation"] - gt["rotation_deg"]) % 360.0
        rot_err = float(raw_rot_err if raw_rot_err <= 180.0 else 360.0 - raw_rot_err)
        
        pred_ds = pred["predicted_scale"] / gt["zoom_ratio"]
        scale_err = float(abs(pred_ds - gt["drift_scale"]))
        
        # Check criteria
        loc_ok = loc_err < 3.0
        rot_ok = rot_err < 0.5
        scale_ok = scale_err < 0.02
        if not (loc_ok and rot_ok and scale_ok):
            status = "FAILURE"
            
    print("==================================================")
    print(f"SAMPLE: {sample_id}")
    print("==================================================")
    print("\nPredicted Center:")
    print(f"X: {pred['predicted_x']:.2f}")
    print(f"Y: {pred['predicted_y']:.2f}")
    print(f"\nPredicted Rotation: {pred['predicted_rotation']:.3f} degrees")
    print(f"Predicted Scale: {pred['predicted_scale']:.3f}")
    
    if has_gt:
        print(f"\nLocalization Error: {loc_err:.2f} px")
        print(f"Rotation Error: {rot_err:.3f} degrees")
        print(f"Scale Error: {scale_err:.4f}")
        
    print(f"\nInference Time: {pred['elapsed_s']:.2f} seconds")
    print(f"\nStatus: {status}\n")

def main():
    print("Initializing Wafer Matcher Demo...")
    matcher = FinalSystemMatcher("configs/final_system_config.json")
    
    samples = [
        ("sample_000", "data/sample_000"),
        ("sample_010", "data/robustness_samples/sample_010"),
        ("sample_020", "data/robustness_samples/sample_020"),
        ("sample_030", "data/robustness_samples/sample_030"),
        ("sample_021", "data/sample_021")
    ]
    
    for sample_id, path in samples:
        run_sample_demo(matcher, sample_id, path)

if __name__ == "__main__":
    main()
