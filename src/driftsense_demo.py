"""
DriftSense sensorless navigation drift recovery terminal demo.
"""
import os
import json
import cv2
from src.final_system import FinalSystemMatcher
from src.drift_recovery import DriftRecoveryModule

def main():
    print("====================================================")
    print("DRIFTSENSE — SENSORLESS NAVIGATION DRIFT RECOVERY")
    print("====================================================")
    
    # STEP 1: Expected inspection site loaded
    expected_x = 450.0
    expected_y = 170.0
    
    print("\nSTEP 1: Expected inspection site loaded")
    print("\nExpected Target:")
    print(f"X = {expected_x:.2f}")
    print(f"Y = {expected_y:.2f}")
    
    # STEP 2: Running wafer pattern localization
    print("\nSTEP 2: Running wafer pattern localization...")
    matcher = FinalSystemMatcher("configs/final_system_config.json")
    
    ref_path = "data/sample_000/reference_image.png"
    srch_path = "data/sample_000/search_image.png"
    
    if not os.path.exists(ref_path) or not os.path.exists(srch_path):
        print("Error: Reference or search image files missing for demo.")
        return
        
    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    srch = cv2.imread(srch_path, cv2.IMREAD_GRAYSCALE)
    
    pred_res = matcher.match(ref, srch)
    
    det_x = pred_res["predicted_x"]
    det_y = pred_res["predicted_y"]
    
    print("\nDetected Target:")
    print(f"X = {det_x:.2f}")
    print(f"Y = {det_y:.2f}")
    print("\nPose:")
    print(f"Rotation = {pred_res['predicted_rotation']:.3f} degrees")
    print(f"Scale = {pred_res['predicted_scale']:.3f}")
    
    # STEP 3: Calculating navigation drift
    print("\nSTEP 3: Calculating navigation drift...")
    
    recovery = DriftRecoveryModule(aligned_max_px=1.0, minor_drift_max_px=5.0)
    drift = recovery.calculate_drift(
        {"x": expected_x, "y": expected_y},
        {"x": det_x, "y": det_y}
    )
    
    dx = drift["dx_pixels"]
    dy = drift["dy_pixels"]
    mag = drift["magnitude_pixels"]
    status = drift["status"]
    
    print(f"\nΔX = {dx:+.2f} pixels")
    print(f"ΔY = {dy:+.2f} pixels")
    print(f"\nDrift Magnitude = {mag:.2f} pixels")
    print(f"\nSTATUS: {status}")
    
    # STEP 4: Recommended coordinate correction
    print("\nSTEP 4: Recommended coordinate correction:")
    print(f"\nMOVE X: {dx:+.2f}")
    print(f"MOVE Y: {dy:+.2f}")
    
    print("\nTARGET RECOVERY COMPLETE")
    print("====================================================")

if __name__ == "__main__":
    main()
