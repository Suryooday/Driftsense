"""
Drift Sense - Evaluation Runner.

Loads all generated dataset samples, executes SIFT and robust template matching
pipelines on each, computes errors, and prints an evaluation summary table
comparing the two methods.
"""

import os
import json
import glob
import cv2
import numpy as np
from src.matching.matcher import Matcher
from src.scoring.evaluator import evaluate_match


def main() -> None:
    """Executes matching and evaluation across all dataset samples."""
    print("Initializing evaluation run...")
    matcher = Matcher()
    
    sample_dirs = sorted(glob.glob("data/sample_*"))
    if not sample_dirs:
        print("No samples found. Please generate the dataset first.")
        return

    # SIFT statistics
    sift_success_count = 0
    sift_pixel_errors = []
    sift_rot_errors = []
    sift_scale_errors = []
    sift_failed = 0

    # Template Matching statistics
    tmpl_success_count = 0
    tmpl_pixel_errors = []
    tmpl_rot_errors = []
    tmpl_scale_errors = []
    tmpl_failed = 0

    print(f"Running evaluation on {len(sample_dirs)} samples...")
    print("-" * 115)
    print(f"{'Sample':<12} | {'GT (X, Y)':<15} | {'SIFT Pred':<15} | {'SIFT Err':<9} | {'Tmpl Pred':<15} | {'Tmpl Err':<9} | {'Tmpl Status':<10}")
    print("-" * 115)

    for sample_dir in sample_dirs:
        sample_name = os.path.basename(sample_dir)
        
        # Load search and reference images
        search_path = os.path.join(sample_dir, "search_image.png")
        ref_path = os.path.join(sample_dir, "reference_image.png")
        gt_path = os.path.join(sample_dir, "ground_truth.json")

        search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)

        with open(gt_path, "r") as f:
            gt = json.load(f)

        gt_coord_str = f"({gt['true_x']:.1f}, {gt['true_y']:.1f})"

        # 1. Evaluate SIFT Matcher
        sift_pred = matcher.classical_match(ref_img, search_img)
        if sift_pred is None:
            sift_failed += 1
            sift_pred_str = "FAILED"
            sift_err_str = "-"
        else:
            sift_metrics = evaluate_match(
                predicted_x=sift_pred["pred_x"],
                predicted_y=sift_pred["pred_y"],
                predicted_rot=sift_pred["pred_rot"],
                predicted_scale=sift_pred["pred_scale"],
                gt=gt
            )
            sift_pred_str = f"({sift_pred['pred_x']:.1f}, {sift_pred['pred_y']:.1f})"
            sift_err_str = f"{sift_metrics['pixel_error']:.2f}px"
            sift_pixel_errors.append(sift_metrics["pixel_error"])
            sift_rot_errors.append(sift_metrics["rotation_error"])
            sift_scale_errors.append(sift_metrics["scale_error"])
            if sift_metrics["success"]:
                sift_success_count += 1

        # 2. Evaluate Template Matcher
        tmpl_pred = matcher.template_match(ref_img, search_img)
        if tmpl_pred is None:
            tmpl_failed += 1
            tmpl_pred_str = "FAILED"
            tmpl_err_str = "-"
            tmpl_status = "FAIL"
        else:
            tmpl_metrics = evaluate_match(
                predicted_x=tmpl_pred["pred_x"],
                predicted_y=tmpl_pred["pred_y"],
                predicted_rot=tmpl_pred["pred_rot"],
                predicted_scale=tmpl_pred["pred_scale"],
                gt=gt
            )
            tmpl_pred_str = f"({tmpl_pred['pred_x']:.1f}, {tmpl_pred['pred_y']:.1f})"
            tmpl_err_str = f"{tmpl_metrics['pixel_error']:.2f}px"
            tmpl_pixel_errors.append(tmpl_metrics["pixel_error"])
            tmpl_rot_errors.append(tmpl_metrics["rotation_error"])
            tmpl_scale_errors.append(tmpl_metrics["scale_error"])
            if tmpl_metrics["success"]:
                tmpl_success_count += 1
                tmpl_status = "SUCCESS"
            else:
                tmpl_status = "FAIL"

        print(f"{sample_name:<12} | {gt_coord_str:<15} | {sift_pred_str:<15} | {sift_err_str:<9} | {tmpl_pred_str:<15} | {tmpl_err_str:<9} | {tmpl_status:<10}")

    print("-" * 115)
    
    # Summary
    num_samples = len(sample_dirs)
    sift_success_rate = (sift_success_count / num_samples) * 100
    tmpl_success_rate = (tmpl_success_count / num_samples) * 100

    print("Evaluation Summary Comparison:")
    print(f"{'Metric':<30} | {'SIFT Feature Matching':<25} | {'Robust Template Matching':<25}")
    print("-" * 88)
    print(f"{'Successful Matches':<30} | {sift_success_count:<3} / {num_samples} ({sift_success_rate:.2f}%) | {tmpl_success_count:<3} / {num_samples} ({tmpl_success_rate:.2f}%)")
    print(f"{'Avg Pixel Location Error':<30} | {np.mean(sift_pixel_errors) if sift_pixel_errors else float('nan'):.3f} px | {np.mean(tmpl_pixel_errors) if tmpl_pixel_errors else float('nan'):.3f} px")
    print(f"{'Avg Rotation Error':<30} | {np.mean(sift_rot_errors) if sift_rot_errors else float('nan'):.3f}° | {np.mean(tmpl_rot_errors) if tmpl_rot_errors else float('nan'):.3f}°")
    print(f"{'Avg Scale Error':<30} | {np.mean(sift_scale_errors) if sift_scale_errors else float('nan'):.4f} | {np.mean(tmpl_scale_errors) if tmpl_scale_errors else float('nan'):.4f}")
    print("-" * 88)


if __name__ == "__main__":
    main()
