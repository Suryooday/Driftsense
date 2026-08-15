"""
DriftSense — Hackathon Scoring Utility.

Evaluates the pattern matcher on any CSV dataset containing:
  search_image_path, reference_image_path, GTx, GTy (where 0,0 is Top-Left)

Publishes:
  - 1px to 5px Accuracy Breakdown
  - 1px to 5px Confusion Matrix / Bucket Table
  - Mean & Median Location Errors
  - Inference Speed Metrics
"""
import os
import sys
import csv
import math
import time
import argparse
import numpy as np
import cv2

from src.final_system import FinalSystemMatcher

def find_column(headers: list, candidates: list) -> str:
    headers_lower = [h.strip().lower() for h in headers]
    for cand in candidates:
        cand_lower = cand.lower()
        if cand_lower in headers_lower:
            idx = headers_lower.index(cand_lower)
            return headers[idx]
    return ""

def evaluate_csv(csv_path: str, output_csv: str = "results/eval_results.csv", config_path: str = "configs/final_system_config.json"):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at '{csv_path}'")
        sys.exit(1)
        
    print(f"Loading dataset CSV: {csv_path}")
    matcher = FinalSystemMatcher(config_path)
    
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        
        col_srch = find_column(headers, ["wide search image path", "search_image_path", "search_path", "search", "search_image"])
        col_ref  = find_column(headers, ["reference image path", "reference_image_path", "ref_path", "reference", "ref_image"])
        col_gtx  = find_column(headers, ["gtx", "gt_x", "true_x", "x_gt", "target_x"])
        col_gty  = find_column(headers, ["gty", "gt_y", "true_y", "y_gt", "target_y"])
        
        if not (col_srch and col_ref and col_gtx and col_gty):
            print(f"Error: Required CSV columns not detected. Found headers: {headers}")
            print("Expected columns: 'search_image_path', 'reference_image_path', 'GTx', 'GTy'")
            sys.exit(1)
            
        for r in reader:
            rows.append({
                "search_path": r[col_srch].strip(),
                "ref_path": r[col_ref].strip(),
                "gt_x": float(r[col_gtx]),
                "gt_y": float(r[col_gty])
            })
            
    print(f"Loaded {len(rows)} image pairs for evaluation.")
    
    results = []
    total_time = 0.0
    
    for i, item in enumerate(rows):
        srch_path = item["search_path"]
        ref_path = item["ref_path"]
        gt_x = item["gt_x"]
        gt_y = item["gt_y"]
        
        if not os.path.exists(srch_path) or not os.path.exists(ref_path):
            print(f"[{i+1}/{len(rows)}] Warning: Image path missing ({srch_path} or {ref_path}). Skipping.")
            continue
            
        srch_img = cv2.imread(srch_path, cv2.IMREAD_GRAYSCALE)
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        
        t0 = time.perf_counter()
        pred = matcher.match(ref_img, srch_img)
        t_elapsed = time.perf_counter() - t0
        total_time += t_elapsed
        
        pred_x = pred["predicted_x"]
        pred_y = pred["predicted_y"]
        
        if pred_x is None or pred_y is None:
            loc_err = 999.0
        else:
            loc_err = math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
            
        results.append({
            "index": i + 1,
            "search_path": srch_path,
            "ref_path": ref_path,
            "gt_x": gt_x,
            "gt_y": gt_y,
            "pred_x": pred_x,
            "pred_y": pred_y,
            "loc_error": loc_err,
            "rotation": pred["predicted_rotation"],
            "scale": pred["predicted_scale"],
            "confidence": pred["confidence_score"],
            "elapsed_s": t_elapsed
        })
        
    N = len(results)
    if N == 0:
        print("No valid pairs evaluated.")
        return
        
    errors = [r["loc_error"] for r in results]
    mean_err = float(np.mean(errors))
    median_err = float(np.median(errors))
    avg_time = total_time / N
    
    # 1px to 5px Accuracy thresholds
    acc_1px = sum(1 for e in errors if e <= 1.0) / N * 100.0
    acc_2px = sum(1 for e in errors if e <= 2.0) / N * 100.0
    acc_3px = sum(1 for e in errors if e <= 3.0) / N * 100.0
    acc_4px = sum(1 for e in errors if e <= 4.0) / N * 100.0
    acc_5px = sum(1 for e in errors if e <= 5.0) / N * 100.0
    
    # Bucket confusion matrix
    b_0_1 = sum(1 for e in errors if e <= 1.0)
    b_1_2 = sum(1 for e in errors if 1.0 < e <= 2.0)
    b_2_3 = sum(1 for e in errors if 2.0 < e <= 3.0)
    b_3_4 = sum(1 for e in errors if 3.0 < e <= 4.0)
    b_4_5 = sum(1 for e in errors if 4.0 < e <= 5.0)
    b_gt5 = sum(1 for e in errors if e > 5.0)
    
    # Print formatted scoring table
    print("\n" + "="*80)
    print("DRIFTSENSE — HACKATHON SCORING UTILITY RESULTS")
    print("="*80)
    print(f"Total Evaluated Image Pairs: {N}")
    print(f"Mean Location Error:        {mean_err:.4f} px")
    print(f"Median Location Error:      {median_err:.4f} px")
    print(f"Average Inference Time:     {avg_time:.4f} s/pair")
    print("-"*80)
    
    print("ACCURACY BREAKDOWN (CM @ 1px - 5px Tolerance):")
    print("  Tolerance   |  Correct (Count)  |  Failed (Count)  |  Accuracy (%)")
    print("  ------------|-------------------|------------------|----------------")
    print(f"   <= 1.0 px  |  {b_0_1:15d}  |  {N - b_0_1:14d}  |  {acc_1px:12.2f} %")
    print(f"   <= 2.0 px  |  {b_0_1+b_1_2:15d}  |  {N - (b_0_1+b_1_2):14d}  |  {acc_2px:12.2f} %")
    print(f"   <= 3.0 px  |  {b_0_1+b_1_2+b_2_3:15d}  |  {N - (b_0_1+b_1_2+b_2_3):14d}  |  {acc_3px:12.2f} %")
    print(f"   <= 4.0 px  |  {b_0_1+b_1_2+b_2_3+b_3_4:15d}  |  {N - (b_0_1+b_1_2+b_2_3+b_3_4):14d}  |  {acc_4px:12.2f} %")
    print(f"   <= 5.0 px  |  {b_0_1+b_1_2+b_2_3+b_3_4+b_4_5:15d}  |  {N - (b_0_1+b_1_2+b_2_3+b_3_4+b_4_5):14d}  |  {acc_5px:12.2f} %")
    print("-"*80)
    
    print("CONFUSION MATRIX / ERROR BUCKET DISTRIBUTION:")
    print("  Bucket Range       |  Count  |  Percentage (%)")
    print("  -------------------|---------|----------------")
    print(f"  [0.0, 1.0] px      |  {b_0_1:5d}  |  {(b_0_1/N*100):12.2f} % (Sub-pixel target)")
    print(f"  (1.0, 2.0] px      |  {b_1_2:5d}  |  {(b_1_2/N*100):12.2f} %")
    print(f"  (2.0, 3.0] px      |  {b_2_3:5d}  |  {(b_2_3/N*100):12.2f} %")
    print(f"  (3.0, 4.0] px      |  {b_3_4:5d}  |  {(b_3_4/N*100):12.2f} %")
    print(f"  (4.0, 5.0] px      |  {b_4_5:5d}  |  {(b_4_5/N*100):12.2f} %")
    print(f"  > 5.0 px (Outlier) |  {b_gt5:5d}  |  {(b_gt5/N*100):12.2f} %")
    print("="*80)
    
    # Save detailed CSV output
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "index", "search_path", "ref_path", "gt_x", "gt_y", "pred_x", "pred_y", "loc_error", "rotation", "scale", "confidence", "elapsed_s"
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDetailed evaluation CSV saved to: {output_csv}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driftsense Hackathon Scoring Utility")
    parser.add_argument("--csv", required=True, help="Path to ground truth CSV file")
    parser.add_argument("--output-csv", default="results/eval_results.csv", help="Path to save output results CSV")
    parser.add_argument("--config", default="configs/final_system_config.json", help="Path to system config")
    
    args = parser.parse_args()
    evaluate_csv(args.csv, args.output_csv, args.config)
