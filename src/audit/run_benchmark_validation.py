"""
Re-runs the classical pose-refined matcher on the 40 frozen benchmark samples
and checks for exact reproducibility.
"""
import os
import json
import glob
import sys
import numpy as np
import cv2

from src.final_system import FinalSystemMatcher

def main():
    print("Running Final Matcher validation on the 40 frozen benchmark samples...")
    matcher = FinalSystemMatcher("configs/final_system_config.json")
    
    benchmark_dir = "data"
    benchmark_paths = sorted([p for p in glob.glob(os.path.join(benchmark_dir, "sample_*")) if os.path.basename(p) < "sample_040"])
    
    predictions = {}
    results = []
    
    for path in benchmark_paths:
        sample_id = os.path.basename(path)
        ref_path = os.path.join(path, "reference_image.png")
        srch_path = os.path.join(path, "search_image.png")
        gt_path = os.path.join(path, "ground_truth.json")
        
        ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        srch = cv2.imread(srch_path, cv2.IMREAD_GRAYSCALE)
        with open(gt_path, "r") as f:
            gt = json.load(f)
            
        pred_res = matcher.match(ref, srch, sample_id=sample_id)
        predictions[sample_id] = pred_res
        
        # Calculate errors
        loc_error = float(np.sqrt((pred_res["predicted_x"] - gt["true_x"])**2 + (pred_res["predicted_y"] - gt["true_y"])**2))
        
        raw_rot_err = abs(pred_res["predicted_rotation"] - gt["rotation_deg"]) % 360.0
        rot_error = float(raw_rot_err if raw_rot_err <= 180.0 else 360.0 - raw_rot_err)
        
        pred_ds = pred_res["predicted_scale"] / gt["zoom_ratio"]
        scale_error = float(abs(pred_ds - gt["drift_scale"]))
        
        loc_ok = loc_error < 3.0
        rot_ok = rot_error < 0.5
        scale_ok = scale_error < 0.02
        all_ok = bool(loc_ok and rot_ok and scale_ok)
        
        results.append({
            "sample_id": sample_id,
            "loc_error": loc_error,
            "rot_error": rot_error,
            "scale_error": scale_error,
            "all_ok": all_ok,
            "elapsed_s": pred_res["elapsed_s"]
        })
        
    # Save predictions
    pred_save_path = "data/final_predictions_benchmark.json"
    with open(pred_save_path, "w") as f:
        json.dump(predictions, f, indent=4)
    print(f"Predictions saved to {pred_save_path}")
    
    # Compute metrics
    loc_errs = [r["loc_error"] for r in results]
    rot_errs = [r["rot_error"] for r in results]
    scale_errs = [r["scale_error"] for r in results]
    elapsed_times = [r["elapsed_s"] for r in results]
    
    success_rate = sum(1 for r in results if r["all_ok"]) / len(results)
    
    metrics = {
        "mean_loc": float(np.mean(loc_errs)),
        "median_loc": float(np.median(loc_errs)),
        "mean_rot": float(np.mean(rot_errs)),
        "mean_scale": float(np.mean(scale_errs)),
        "success_rate": float(success_rate),
        "mean_time": float(np.mean(elapsed_times)),
        "raw_results": results
    }
    
    metrics_save_path = "reports/final_freeze/benchmark_metrics.json"
    with open(metrics_save_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_save_path}")
    
    # Verify values against baseline
    expected_success = 0.975
    expected_mean_loc = 0.5425
    expected_median_loc = 0.5280
    expected_mean_rot = 0.0910
    expected_mean_scale = 0.00367
    
    print("\n==================================================")
    print("REPRODUCIBILITY CHECK VS BASELINE")
    print("==================================================")
    print(f"Metric                 | Reproduced   | Expected")
    print(f"Success Rate (%)       | {success_rate * 100.0:<12.1f} | {expected_success * 100.0:<12.1f}")
    print(f"Mean Loc Error (px)    | {metrics['mean_loc']:<12.4f} | {expected_mean_loc:<12.4f}")
    print(f"Median Loc Error (px)  | {metrics['median_loc']:<12.4f} | {expected_median_loc:<12.4f}")
    print(f"Mean Rot Error (°)     | {metrics['mean_rot']:<12.4f} | {expected_mean_rot:<12.4f}")
    print(f"Mean Scale Error       | {metrics['mean_scale']:<12.5f} | {expected_mean_scale:<12.5f}")
    print("==================================================")
    
    reproduced = True
    if abs(success_rate - expected_success) > 1e-5:
        print("[FAIL] Success rate discrepancy!")
        reproduced = False
    if abs(metrics["mean_loc"] - expected_mean_loc) > 1e-2:
        print("[FAIL] Mean location error discrepancy!")
        reproduced = False
    if abs(metrics["mean_rot"] - expected_mean_rot) > 1e-2:
        print("[FAIL] Mean rotation error discrepancy!")
        reproduced = False
    if abs(metrics["mean_scale"] - expected_mean_scale) > 1e-3:
        print("[FAIL] Mean scale error discrepancy!")
        reproduced = False
        
    if reproduced:
        print("  [SUCCESS] Matcher output successfully reproduced matching all benchmark targets!")
    else:
        print("  [ERROR] Discrepancy found! Halting execution.")
        sys.exit(1)

if __name__ == "__main__":
    main()
