"""
Drift-Sense Phase 2 — Calibration Scoring Harness (score.py).

Evaluates the Naive Baseline Matcher on the generated dataset and computes:
- Tiered credit score based on localization error (<=1px, <=2px, <=3px, <=5px).
- Overall mean credit on present pairs (Target band: 0.30 - 0.55).
- Median center error and mean credit broken down by set (A, B, C, D) and severity level (1-4).
- Severity monotonicity check (guaranteeing physical difficulty ladder).
- Present and absent peak distribution, separation gap, and presence rejection metrics (Precision, Recall, F1).
- Outputs findings to baseline_calibration.txt.
"""

import os
import sys
import csv
import math
import argparse
import time
from typing import Dict, List, Any
import numpy as np

from baseline import NaiveBaselineMatcher


def compute_credit(error_px: float) -> float:
    if error_px <= 1.0:
        return 1.00
    elif error_px <= 2.0:
        return 0.80
    elif error_px <= 3.0:
        return 0.60
    elif error_px <= 5.0:
        return 0.40
    else:
        return 0.00


def run_evaluation(data_dir: str = "output", max_pairs: int = None) -> Dict[str, Any]:
    gt_csv_path = os.path.join(data_dir, "ground_truth.csv")
    pairs_csv_path = os.path.join(data_dir, "pairs.csv")
    manifest_csv_path = os.path.join(data_dir, "manifest.csv")

    if not os.path.exists(gt_csv_path) or not os.path.exists(pairs_csv_path):
        raise FileNotFoundError(f"Missing dataset files in {data_dir}. Expected ground_truth.csv and pairs.csv")

    # Load ground truth
    gt_records = {}
    with open(gt_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            gt_records[r["pair_id"]] = {
                "present": int(r["present"]),
                "x": float(r["x"]),
                "y": float(r["y"]),
                "theta": float(r["theta"]),
                "scale": float(r["scale"])
            }

    # Load pairs
    pairs = []
    with open(pairs_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pairs.append(r)

    # Load manifest if available for set / severity metadata
    manifest_records = {}
    if os.path.exists(manifest_csv_path):
        with open(manifest_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                manifest_records[r["pair_id"]] = r

    if max_pairs:
        pairs = pairs[:max_pairs]

    print(f"Starting Baseline Calibration Harness on {len(pairs)} pairs in '{data_dir}'...")
    matcher = NaiveBaselineMatcher()

    # Tracking structures
    set_metrics = {
        "A": {"credits": [], "errors": [], "peaks": []},
        "B": {"credits": [], "errors": [], "peaks": []},
        "C": {"credits": [], "errors": [], "peaks": []},
        "D": {"credits": [], "errors": [], "peaks": []}
    }
    severity_metrics = {
        1: {"credits": [], "errors": []},
        2: {"credits": [], "errors": []},
        3: {"credits": [], "errors": []},
        4: {"credits": [], "errors": []}
    }

    present_peaks = []
    absent_peaks = []
    all_present_credits = []
    all_present_errors = []

    # Detection confusion matrix
    tp = fp = tn = fn = 0
    per_pair_results = []

    t0_total = time.perf_counter()

    for idx, p in enumerate(pairs):
        pid = p["pair_id"]
        srch_path = p["search_path"]
        ref_path = p["reference_path"]

        # Ensure path handles relative / absolute
        if not os.path.isabs(srch_path):
            srch_path = os.path.join(data_dir, srch_path)
        if not os.path.isabs(ref_path):
            ref_path = os.path.join(data_dir, ref_path)

        gt = gt_records[pid]
        meta = manifest_records.get(pid, {})
        set_name = meta.get("set", "A" if gt["present"] == 1 else "C")
        sev_lvl = int(meta.get("severity_level", 1))

        t0_pair = time.perf_counter()
        import cv2
        srch_img = cv2.imread(srch_path, cv2.IMREAD_UNCHANGED)
        ref_img = cv2.imread(ref_path, cv2.IMREAD_UNCHANGED)
        pred = matcher.match(ref_img, srch_img)
        elapsed_pair = time.perf_counter() - t0_pair

        gt_present = gt["present"]
        pred_present = pred["predicted_present"]
        peak_score = pred["confidence_score"]

        credit = 0.0
        center_err = None

        if gt_present == 1:
            present_peaks.append(peak_score)
            if pred_present == 1:
                tp += 1
                center_err = float(math.hypot(pred["predicted_x"] - gt["x"], pred["predicted_y"] - gt["y"]))
                credit = compute_credit(center_err)
            else:
                fn += 1
                # Still calculate error to the peak found even if below presence threshold
                center_err = float(math.hypot(pred["predicted_x"] - gt["x"], pred["predicted_y"] - gt["y"]))
                credit = 0.00

            all_present_credits.append(credit)
            all_present_errors.append(center_err)

            if set_name in set_metrics:
                set_metrics[set_name]["credits"].append(credit)
                set_metrics[set_name]["errors"].append(center_err)
                set_metrics[set_name]["peaks"].append(peak_score)

            if set_name == "B" and sev_lvl in severity_metrics:
                severity_metrics[sev_lvl]["credits"].append(credit)
                severity_metrics[sev_lvl]["errors"].append(center_err)

        else:
            absent_peaks.append(peak_score)
            if pred_present == 0:
                tn += 1
                credit = 1.00
            else:
                fp += 1
                credit = 0.00

            if set_name in set_metrics:
                set_metrics[set_name]["credits"].append(credit)
                set_metrics[set_name]["peaks"].append(peak_score)

        per_pair_results.append({
            "pair_id": pid,
            "set": set_name,
            "gt_present": gt_present,
            "pred_present": pred_present,
            "peak_score": peak_score,
            "center_err": center_err,
            "credit": credit,
            "elapsed_s": elapsed_pair
        })

    elapsed_total = time.perf_counter() - t0_total

    # Compute aggregate statistics
    mean_present_credit = float(np.mean(all_present_credits)) if all_present_credits else 0.0
    median_present_error = float(np.median(all_present_errors)) if all_present_errors else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    min_pres_peak = min(present_peaks) if present_peaks else 0.0
    max_pres_peak = max(present_peaks) if present_peaks else 0.0
    min_abs_peak = min(absent_peaks) if absent_peaks else 0.0
    max_abs_peak = max(absent_peaks) if absent_peaks else 0.0
    separation_gap = min_pres_peak - max_abs_peak

    # Format calibration report text
    report_lines = []
    report_lines.append("================================================================================")
    report_lines.append("                 DRIFT-SENSE PHASE 2 — BASELINE CALIBRATION REPORT               ")
    report_lines.append("================================================================================")
    report_lines.append(f"Total Evaluated Pairs: {len(pairs)} (Total Runtime: {elapsed_total:.2f}s, Avg: {elapsed_total/len(pairs):.3f}s/pair)")
    report_lines.append(f"Target Present Credit Band: [0.30, 0.55]")
    report_lines.append(f"Achieved Present Mean Credit: {mean_present_credit:.4f} " + ("🎯 IN TARGET BAND" if 0.30 <= mean_present_credit <= 0.55 else "⚠️ OUTSIDE BAND"))
    report_lines.append(f"Median Present Center Error: {median_present_error:.2f} px")
    report_lines.append("--------------------------------------------------------------------------------")
    report_lines.append(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    report_lines.append(f"Presence Detection -> Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    report_lines.append(f"Present Peak Range: [{min_pres_peak:.4f}, {max_pres_peak:.4f}]")
    report_lines.append(f"Absent Peak Range:  [{min_abs_peak:.4f}, {max_abs_peak:.4f}]")
    report_lines.append(f"Separation Gap (min_pres - max_abs): {separation_gap:.4f} (" + ("Desirable negative overlap" if separation_gap < 0 else "Cleanly separated") + ")")
    report_lines.append("--------------------------------------------------------------------------------")
    report_lines.append("BREAKDOWN BY SET:")
    for s_name in ["A", "B", "C", "D"]:
        sm = set_metrics[s_name]
        mean_c = np.mean(sm["credits"]) if sm["credits"] else 0.0
        med_e = np.median(sm["errors"]) if sm["errors"] else 0.0
        mean_p = np.mean(sm["peaks"]) if sm["peaks"] else 0.0
        report_lines.append(f"  Set {s_name}: Count={len(sm['credits'])}, Mean Credit={mean_c:.4f}, Median Error={med_e:.2f} px, Mean Peak={mean_p:.4f}")

    report_lines.append("--------------------------------------------------------------------------------")
    report_lines.append("SET B SEVERITY MONOTONICITY AUDIT:")
    is_monotone = True
    prev_err = -1.0
    for lvl in [1, 2, 3, 4]:
        svm = severity_metrics[lvl]
        mean_c = np.mean(svm["credits"]) if svm["credits"] else 0.0
        med_e = np.median(svm["errors"]) if svm["errors"] else 0.0
        report_lines.append(f"  Severity {lvl}: Count={len(svm['credits'])}, Mean Credit={mean_c:.4f}, Median Error={med_e:.2f} px")
        if med_e < prev_err:
            is_monotone = False
        prev_err = med_e

    report_lines.append(f"Severity Monotonicity Status: {'✅ STRICTLY MONOTONIC' if is_monotone else '⚠️ NON-MONOTONIC'}")
    report_lines.append("================================================================================")

    full_report_text = "\n".join(report_lines)
    print("\n" + full_report_text)

    calib_out_path = os.path.join(data_dir, "baseline_calibration.txt")
    with open(calib_out_path, "w", encoding="utf-8") as f:
        f.write(full_report_text)
    print(f"\nCalibration report written to {calib_out_path}")

    return {
        "mean_present_credit": mean_present_credit,
        "median_present_error": median_present_error,
        "is_monotone": is_monotone,
        "separation_gap": separation_gap,
        "report_text": full_report_text
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drift-Sense Phase 2 Calibration Harness")
    parser.add_argument("--data-dir", default="output", help="Directory containing dataset and CSVs")
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional maximum pairs to evaluate")
    args = parser.parse_args()

    run_evaluation(args.data_dir, args.max_pairs)
