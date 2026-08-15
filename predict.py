"""
DriftSense — Universal Prediction & Evaluation Entry Point.

Supports two operational modes:

1. Batch CSV Mode (Judges Evaluation):
   python3 predict.py --input test_dataset.csv --output predictions.csv

   - Accepts any CSV with flexible headers (e.g. search_image_path, reference_image_path, GTx, GTy).
   - Runs pattern localization model on every pair.
   - Saves predictions to CSV.
   - If Ground Truth (GTx, GTy) is included in the CSV, prints accuracy breakdown & 1px-5px Confusion Matrix.

2. Single Pair Mode:
   python3 predict.py --reference ref.png --search search.png --output result.json
"""

import os
import sys
import csv
import math
import time
import argparse
import json
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

def process_single_pair(matcher: FinalSystemMatcher, ref_path: str, srch_path: str, output_path: str):
    if not os.path.exists(ref_path):
        print(f"Error: Reference image not found at {ref_path}")
        sys.exit(1)
    if not os.path.exists(srch_path):
        print(f"Error: Search image not found at {srch_path}")
        sys.exit(1)

    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    srch_img = cv2.imread(srch_path, cv2.IMREAD_GRAYSCALE)

    pred = matcher.match(ref_img, srch_img)

    output_data = {
        "predicted_center": {
            "x": round(pred["predicted_x"], 4) if pred["predicted_x"] is not None else None,
            "y": round(pred["predicted_y"], 4) if pred["predicted_y"] is not None else None
        },
        "rotation_degrees": round(pred["predicted_rotation"], 4) if pred["predicted_rotation"] is not None else None,
        "scale": round(pred["predicted_scale"], 4) if pred["predicted_scale"] is not None else None,
        "confidence_score": round(pred["confidence_score"], 4),
        "inference_time_seconds": round(pred["elapsed_s"], 4)
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"\nPrediction saved to: {output_path}")
    print(json.dumps(output_data, indent=2))

def process_csv(matcher: FinalSystemMatcher, input_csv: str, output_csv: str):
    if not os.path.exists(input_csv):
        print(f"Error: Input CSV file not found at '{input_csv}'")
        sys.exit(1)

    print(f"Loading evaluation dataset CSV: {input_csv}")

    rows = []
    headers = []
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        col_srch = find_column(headers, ["wide search image path", "search_image_path", "search_path", "search", "search_image"])
        col_ref  = find_column(headers, ["reference image path", "reference_image_path", "ref_path", "reference", "ref_image"])
        col_gtx  = find_column(headers, ["gtx", "gt_x", "true_x", "x_gt", "target_x"])
        col_gty  = find_column(headers, ["gty", "gt_y", "true_y", "y_gt", "target_y"])

        if not (col_srch and col_ref):
            print(f"Error: Could not detect image path columns in CSV header: {headers}")
            print("Supported search column names: 'search_image_path', 'wide search image path', 'search_path', 'search'")
            print("Supported reference column names: 'reference_image_path', 'ref_path', 'reference'")
            sys.exit(1)

        for r in reader:
            item = {
                "search_path": r[col_srch].strip(),
                "ref_path": r[col_ref].strip()
            }
            if col_gtx and col_gty and r[col_gtx] and r[col_gty]:
                try:
                    item["gt_x"] = float(r[col_gtx])
                    item["gt_y"] = float(r[col_gty])
                except ValueError:
                    pass
            rows.append(item)

    print(f"Running model inference on {len(rows)} image pairs...")

    output_rows = []
    total_time = 0.0
    errors = []

    for i, item in enumerate(rows):
        srch_path = item["search_path"]
        ref_path = item["ref_path"]

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

        out_item = {
            "index": i + 1,
            "search_image_path": srch_path,
            "reference_image_path": ref_path,
            "predicted_x": round(pred_x, 4) if pred_x is not None else None,
            "predicted_y": round(pred_y, 4) if pred_y is not None else None,
            "predicted_rotation": round(pred["predicted_rotation"], 4) if pred["predicted_rotation"] is not None else None,
            "predicted_scale": round(pred["predicted_scale"], 4) if pred["predicted_scale"] is not None else None,
            "confidence_score": round(pred["confidence_score"], 4),
            "elapsed_s": round(t_elapsed, 4)
        }

        if "gt_x" in item and "gt_y" in item:
            gt_x = item["gt_x"]
            gt_y = item["gt_y"]
            out_item["GTx"] = gt_x
            out_item["GTy"] = gt_y
            if pred_x is not None and pred_y is not None:
                loc_err = math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
            else:
                loc_err = 999.0
            out_item["loc_error"] = round(loc_err, 4)
            errors.append(loc_err)

        output_rows.append(out_item)

    # Write output CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    fieldnames = list(output_rows[0].keys()) if output_rows else []
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nModel predictions successfully saved to: {output_csv}")

    # If ground truth was present, print accuracy stats
    N = len(errors)
    if N > 0:
        mean_err = float(np.mean(errors))
        median_err = float(np.median(errors))
        avg_time = total_time / N

        b_0_1 = sum(1 for e in errors if e <= 1.0)
        b_1_2 = sum(1 for e in errors if 1.0 < e <= 2.0)
        b_2_3 = sum(1 for e in errors if 2.0 < e <= 3.0)
        b_3_4 = sum(1 for e in errors if 3.0 < e <= 4.0)
        b_4_5 = sum(1 for e in errors if 4.0 < e <= 5.0)
        b_gt5 = sum(1 for e in errors if e > 5.0)

        print("\n" + "="*80)
        print("MODEL EVALUATION ACCURACY REPORT")
        print("="*80)
        print(f"Total Evaluated Image Pairs: {N}")
        print(f"Mean Location Error:        {mean_err:.4f} px")
        print(f"Median Location Error:      {median_err:.4f} px")
        print(f"Average Inference Time:     {avg_time:.4f} s/pair")
        print("-"*80)
        print("ACCURACY BREAKDOWN (CM @ 1px - 5px Tolerance):")
        print(f"   <= 1.0 px  :  {b_0_1/N*100:6.2f} % ({b_0_1}/{N})")
        print(f"   <= 2.0 px  :  {(b_0_1+b_1_2)/N*100:6.2f} % ({b_0_1+b_1_2}/{N})")
        print(f"   <= 3.0 px  :  {(b_0_1+b_1_2+b_2_3)/N*100:6.2f} % ({b_0_1+b_1_2+b_2_3}/{N})")
        print(f"   <= 4.0 px  :  {(b_0_1+b_1_2+b_2_3+b_3_4)/N*100:6.2f} % ({b_0_1+b_1_2+b_2_3+b_3_4}/{N})")
        print(f"   <= 5.0 px  :  {(b_0_1+b_1_2+b_2_3+b_3_4+b_4_5)/N*100:6.2f} % ({b_0_1+b_1_2+b_2_3+b_3_4+b_4_5}/{N})")
        print("="*80)

def main():
    parser = argparse.ArgumentParser(description="DriftSense Model Prediction & Evaluation script.")
    parser.add_argument("--input", "--csv", help="Path to input test CSV file")
    parser.add_argument("--output", default="results/predictions.csv", help="Path to save predictions output")
    parser.add_argument("--reference", help="Path to single reference image (PNG)")
    parser.add_argument("--search", help="Path to single search image (PNG)")
    parser.add_argument("--config", default="configs/final_system_config.json", help="Path to system config")

    args = parser.parse_args()

    matcher = FinalSystemMatcher(config_path=args.config)

    if args.input:
        process_csv(matcher, args.input, args.output)
    elif args.reference and args.search:
        out_path = args.output if args.output != "results/predictions.csv" else "results/prediction.json"
        process_single_pair(matcher, args.reference, args.search, out_path)
    else:
        print("Usage error: Please provide either --input test_dataset.csv OR --reference ref.png --search search.png")
        sys.exit(1)

if __name__ == "__main__":
    main()
