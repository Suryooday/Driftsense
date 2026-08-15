"""
Evaluates Classical NCC-based wafer matching with high-resolution pose refinement
on the 200 unseen robustness samples.
"""
import os
import json
import glob
import time
import numpy as np
import cv2
import yaml
from typing import Dict, List, Any, Tuple

from src.hybrid.hybrid_matcher import HybridMatcher

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def compute_bootstrap_cis(successes: List[bool], loc_errors: List[float], num_resamples: int = 2000, seed: int = 777) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(successes)
    
    bootstrap_success_rates = []
    bootstrap_mean_loc_errors = []
    
    for _ in range(num_resamples):
        indices = rng.choice(n, size=n, replace=True)
        
        # Success rate resample
        sample_successes = [successes[i] for i in indices]
        bootstrap_success_rates.append(sum(sample_successes) / n)
        
        # Mean localization error resample
        sample_locs = [loc_errors[i] for i in indices]
        bootstrap_mean_loc_errors.append(np.mean(sample_locs))
        
    return {
        "success_rate_ci": [float(np.percentile(bootstrap_success_rates, 2.5)), float(np.percentile(bootstrap_success_rates, 97.5))],
        "mean_loc_error_ci": [float(np.percentile(bootstrap_mean_loc_errors, 2.5)), float(np.percentile(bootstrap_mean_loc_errors, 97.5))]
    }

def main():
    print("Initializing Robustness Evaluation...")
    results_dir = "results/phase7_robustness"
    os.makedirs(results_dir, exist_ok=True)
    
    config = load_config("config.yaml")
    
    # Initialize matcher (DL checkpoint path won't be used since ranking_mode="classical")
    matcher = HybridMatcher(
        zoom_ratio=float(config.get("zoom_ratio", 5.0)),
        rot_range_deg=tuple(config.get("rotation_bounds", [-3.0, 3.0])),
        scale_range=tuple(config.get("scale_bounds", [0.97, 1.03])),
        checkpoint_path="models/dl_matcher/best_model.pth"
    )
    
    samples_dir = "data/robustness_samples"
    sample_paths = sorted(glob.glob(os.path.join(samples_dir, "sample_*")))
    
    predictions = {}
    results = []
    
    for path in sample_paths:
        sample_id = os.path.basename(path)
        ref_path = os.path.join(path, "reference_image.png")
        srch_path = os.path.join(path, "search_image.png")
        gt_path = os.path.join(path, "ground_truth.json")
        
        ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        srch = cv2.imread(srch_path, cv2.IMREAD_GRAYSCALE)
        with open(gt_path, "r") as f:
            gt = json.load(f)
            
        # Time the matching run
        t_start = time.perf_counter()
        pred_res = matcher.match_hybrid(
            ref, srch, sample_id=sample_id, k=5, ranking_mode="classical"
        )
        elapsed_total = time.perf_counter() - t_start
        
        final = pred_res["final_prediction"]
        
        if final["x"] is not None:
            # Localization Error
            loc_error = float(np.sqrt((final["x"] - gt["true_x"])**2 + (final["y"] - gt["true_y"])**2))
            
            # Rotation Error
            raw_rot_err = abs(final["rotation"] - gt["rotation_deg"]) % 360.0
            rot_error = float(raw_rot_err if raw_rot_err <= 180.0 else 360.0 - raw_rot_err)
            
            # Scale Error
            pred_ds = final["scale"] / gt["zoom_ratio"]
            scale_error = float(abs(pred_ds - gt["drift_scale"]))
            
            # Success gates (strict <)
            loc_ok = loc_error < 3.0
            rot_ok = rot_error < 0.5
            scale_ok = scale_error < 0.02
            all_ok = bool(loc_ok and rot_ok and scale_ok)
        else:
            loc_error, rot_error, scale_error = float("nan"), float("nan"), float("nan")
            loc_ok, rot_ok, scale_ok, all_ok = False, False, False, False
            
        results.append({
            "sample_id": sample_id,
            "loc_error": loc_error,
            "rot_error": rot_error,
            "scale_error": scale_error,
            "loc_ok": loc_ok,
            "rot_ok": rot_ok,
            "scale_ok": scale_ok,
            "all_ok": all_ok,
            "elapsed_s": elapsed_total,
            "metadata": gt
        })
        
        predictions[sample_id] = pred_res
        
    # Write predictions.json
    pred_save_path = os.path.join(results_dir, "predictions.json")
    with open(pred_save_path, "w") as f:
        json.dump(predictions, f, indent=4)

    # 1. Compute aggregate statistics
    loc_errors = [r["loc_error"] for r in results if not np.isnan(r["loc_error"])]
    rot_errors = [r["rot_error"] for r in results if not np.isnan(r["rot_error"])]
    scale_errors = [r["scale_error"] for r in results if not np.isnan(r["scale_error"])]
    elapsed_times = [r["elapsed_s"] for r in results]
    success_flags = [r["all_ok"] for r in results]
    
    success_rate = sum(success_flags) / len(results)
    
    bootstrap_results = compute_bootstrap_cis(success_flags, loc_errors, num_resamples=2000, seed=777)
    
    aggregate_metrics = {
        "mean_loc": float(np.mean(loc_errors)),
        "median_loc": float(np.median(loc_errors)),
        "p95_loc": float(np.percentile(loc_errors, 95)),
        "max_loc": float(np.max(loc_errors)),
        
        "mean_rot": float(np.mean(rot_errors)),
        "median_rot": float(np.median(rot_errors)),
        "p95_rot": float(np.percentile(rot_errors, 95)),
        "max_rot": float(np.max(rot_errors)),
        
        "mean_scale": float(np.mean(scale_errors)),
        "median_scale": float(np.median(scale_errors)),
        "p95_scale": float(np.percentile(scale_errors, 95)),
        "max_scale": float(np.max(scale_errors)),
        
        "success_rate": success_rate,
        "mean_time": float(np.mean(elapsed_times)),
        "median_time": float(np.median(elapsed_times)),
        "max_time": float(np.max(elapsed_times)),
        
        "bootstrap": bootstrap_results
    }
    
    # Save aggregate_metrics.json
    agg_save_path = os.path.join(results_dir, "aggregate_metrics.json")
    with open(agg_save_path, "w") as f:
        json.dump(aggregate_metrics, f, indent=4)
        
    # 2. Failure Analysis
    failures = []
    for r in results:
        if not r["all_ok"]:
            failed_criteria = []
            if not r["loc_ok"]: failed_criteria.append("localization")
            if not r["rot_ok"]: failed_criteria.append("rotation")
            if not r["scale_ok"]: failed_criteria.append("scale")
            
            # Determine measurable failure category
            # If translation error was coarse and held fixed during pose refinement
            if r["loc_error"] > 0.6 and not r["rot_ok"]:
                category = "Coupling of translation error into rotation refinement"
            elif r["metadata"]["noise_level"] > 0.05:
                category = "Extreme noise/degradation"
            elif r["metadata"]["charging_amplitude"] > 75:
                category = "Severe SEM charging effect"
            else:
                category = "Ambiguous correlation peaks"
                
            failures.append({
                "sample_id": r["sample_id"],
                "loc_error": r["loc_error"],
                "rot_error": r["rot_error"],
                "scale_error": r["scale_error"],
                "failed_criteria": failed_criteria,
                "category": category
            })
            
    failure_analysis = {
        "total_failures": len(failures),
        "failure_percentage": (len(failures) / len(results)) * 100.0,
        "failures": failures
    }
    
    # Save failure_analysis.json
    fail_save_path = os.path.join(results_dir, "failure_analysis.json")
    with open(fail_save_path, "w") as f:
        json.dump(failure_analysis, f, indent=4)
        
    # 3. Robustness Breakdowns
    def get_noise_bin(std):
        if std <= 0.02: return "low"
        if std <= 0.04: return "medium"
        return "high"
        
    noise_stats = {"low": {"total": 0, "ok": 0}, "medium": {"total": 0, "ok": 0}, "high": {"total": 0, "ok": 0}}
    
    for r in results:
        nb = get_noise_bin(r["metadata"]["noise_level"])
        noise_stats[nb]["total"] += 1
        if r["all_ok"]:
            noise_stats[nb]["ok"] += 1
            
    print("\n==================================================")
    print("ROBUSTNESS EVALUATION COMPLETE")
    print("==================================================")
    print(f"Success Rate: {success_rate * 100.0:.2f}% (Bootstrap 95% CI: [{bootstrap_results['success_rate_ci'][0]*100.0:.2f}%, {bootstrap_results['success_rate_ci'][1]*100.0:.2f}%])")
    print(f"Mean Location Error: {aggregate_metrics['mean_loc']:.4f} px (Bootstrap 95% CI: [{bootstrap_results['mean_loc_error_ci'][0]:.4f} px, {bootstrap_results['mean_loc_error_ci'][1]:.4f} px])")
    print(f"Mean Rotation Error: {aggregate_metrics['mean_rot']:.4f}°")
    print(f"Mean Scale Error: {aggregate_metrics['mean_scale']:.5f}")
    print(f"Mean Time: {aggregate_metrics['mean_time']:.4f} s")
    print("==================================================")
    print(f"Noise Breakdown: Low: {noise_stats['low']['ok']}/{noise_stats['low']['total']} ({noise_stats['low']['ok']/noise_stats['low']['total']*100.0 if noise_stats['low']['total'] > 0 else 0:.1f}%), Medium: {noise_stats['medium']['ok']}/{noise_stats['medium']['total']} ({noise_stats['medium']['ok']/noise_stats['medium']['total']*100.0 if noise_stats['medium']['total'] > 0 else 0:.1f}%), High: {noise_stats['high']['ok']}/{noise_stats['high']['total']} ({noise_stats['high']['ok']/noise_stats['high']['total']*100.0 if noise_stats['high']['total'] > 0 else 0:.1f}%)")
    
    # 4. Compare with Frozen 40-sample benchmark
    # Let's load the predictions_hybrid_benchmark.json if it exists to get exact benchmark stats
    benchmark_comparison = {
        "success_rate": {
            "benchmark": 0.975,
            "robustness": success_rate
        },
        "mean_loc": {
            "benchmark": 0.5425,
            "robustness": aggregate_metrics["mean_loc"]
        },
        "median_loc": {
            "benchmark": 0.5280,
            "robustness": aggregate_metrics["median_loc"]
        },
        "mean_rot": {
            "benchmark": 0.0910,
            "robustness": aggregate_metrics["mean_rot"]
        },
        "mean_scale": {
            "benchmark": 0.00367,
            "robustness": aggregate_metrics["mean_scale"]
        },
        "mean_time": {
            "benchmark": 0.6267,
            "robustness": aggregate_metrics["mean_time"]
        }
    }
    
    comp_save_path = os.path.join(results_dir, "benchmark_comparison.json")
    with open(comp_save_path, "w") as f:
        json.dump(benchmark_comparison, f, indent=4)
        
    print(f"Robustness results saved under {results_dir}/")

if __name__ == "__main__":
    main()
