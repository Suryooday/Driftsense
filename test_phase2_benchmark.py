import os
import sys
import csv
import time
import math
import numpy as np
import cv2
from pathlib import Path

# Add project root to python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.final_system import FinalSystemMatcher

def load_benchmark_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "pair_id": r["pair_id"],
                "search_image_path": r["search_image_path"],
                "reference_image_path": r["reference_image_path"],
                "GTx": float(r["GTx"]),
                "GTy": float(r["GTy"]),
                "GT_theta": float(r["GT_theta"]),
                "GT_scale": float(r["GT_scale"]),
                "GT_found": int(r["GT_found"]),
                "set": r["set"],
                "style": r["style"]
            })
    return rows

def main():
    print("Initializing Wafer Alignment Benchmark...")
    csv_path = "data/phase2_test_data/ground_truth.csv"
    if not os.path.exists(csv_path):
        print(f"Error: Benchmark ground truth not found at {csv_path}. Please run generate_phase2_benchmark.py first.")
        sys.exit(1)
        
    rows = load_benchmark_csv(csv_path)
    print(f"Loaded {len(rows)} image pairs.")
    
    matcher = FinalSystemMatcher()
    
    # We will track metrics by set: 'A', 'B', 'C', 'D'
    set_stats = {
        "A": {"total": 0, "correct": 0, "runtimes": [], "errors": []},
        "B": {"total": 0, "correct": 0, "runtimes": [], "errors": []},
        "C": {"total": 0, "correct": 0, "runtimes": [], "errors": []},
        "D": {"total": 0, "correct": 0, "runtimes": [], "errors": []}
    }
    
    print("\nStarting execution on 200 pairs...")
    t_start = time.perf_counter()
    
    for idx, row in enumerate(rows):
        pid = row["pair_id"]
        set_type = row["set"]
        style = row["style"]
        srch_path = row["search_image_path"]
        ref_path = row["reference_image_path"]
        gt_found = row["GT_found"]
        gt_x = row["GTx"]
        gt_y = row["GTy"]
        
        # Load images
        srch_img = cv2.imread(srch_path, cv2.IMREAD_GRAYSCALE)
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        
        t0 = time.perf_counter()
        pred = matcher.match(ref_img, srch_img)
        elapsed = time.perf_counter() - t0
        
        # Track stats
        stats = set_stats[set_type]
        stats["total"] += 1
        stats["runtimes"].append(elapsed)
        
        pred_found = pred["found"]
        pred_x = pred["predicted_x"]
        pred_y = pred["predicted_y"]
        score = pred["confidence_score"]
        
        is_correct = False
        err = None
        
        if gt_found == 1:
            if pred_found == 1:
                err = math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
                stats["errors"].append(err)
                if err <= 5.0:
                    is_correct = True
            else:
                # False Negative
                pass
        else: # gt_found == 0
            if pred_found == 0:
                is_correct = True # Correct Rejection
            else:
                # False Positive
                pass
                
        if is_correct:
            stats["correct"] += 1
            
        # Logging
        if gt_found == 1:
            err_str = f"Loc Error: {err:.4f} px" if err is not None else "Not Found"
            print(f"[{idx+1}/200] Set {set_type} ({style}) | Pair: {pid} | GT Found: 1 | Pred Found: {pred_found} | Score: {score:.4f} | {err_str} | Time: {elapsed:.2f}s")
        else:
            print(f"[{idx+1}/200] Set {set_type} ({style}) | Pair: {pid} | GT Found: 0 | Pred Found: {pred_found} | Score: {score:.4f} | Time: {elapsed:.2f}s")
            
    t_total = time.perf_counter() - t_start
    print(f"\nBenchmark completed in {t_total:.2f} seconds.")
    
    # Generate markdown table report
    print("\nGenerating Results Summary...")
    
    headers = ["Set Name", "Description", "Total Pairs", "Correct Poses / Rejections", "Success Rate (%)", "Median Error (px)", "Max Error (px)", "Average Time (ms)"]
    table_rows = []
    
    desc_map = {
        "A": "Nominal pose (Full [8,12]x & +-5 deg scale/rot)",
        "B": "Degraded (Charging, distortion, elevated noise, blur)",
        "C": "Absent (Different die region of same arch)",
        "D": "Optical microscope analogue (RGB 3-channel)"
    }
    
    total_pairs = 0
    total_correct = 0
    all_runtimes = []
    all_errors = []
    
    for set_type in ["A", "B", "C", "D"]:
        stats = set_stats[set_type]
        total = stats["total"]
        correct = stats["correct"]
        runtimes = stats["runtimes"]
        errors = stats["errors"]
        
        total_pairs += total
        total_correct += correct
        all_runtimes.extend(runtimes)
        all_errors.extend(errors)
        
        success_rate = (correct / total) * 100 if total > 0 else 0.0
        median_err = np.median(errors) if errors else 0.0
        max_err = np.max(errors) if errors else 0.0
        avg_time_ms = np.mean(runtimes) * 1000.0 if runtimes else 0.0
        
        table_rows.append([
            f"Set {set_type}",
            desc_map[set_type],
            str(total),
            str(correct),
            f"{success_rate:.2f}%",
            f"{median_err:.4f}" if set_type != "C" else "N/A",
            f"{max_err:.4f}" if set_type != "C" else "N/A",
            f"{avg_time_ms:.1f}"
        ])
        
    # Overall summary row
    overall_success = (total_correct / total_pairs) * 100
    overall_median_err = np.median(all_errors) if all_errors else 0.0
    overall_max_err = np.max(all_errors) if all_errors else 0.0
    overall_avg_time_ms = np.mean(all_runtimes) * 1000.0
    
    table_rows.append([
        "**Overall**",
        "**Combined Benchmark**",
        f"**{total_pairs}**",
        f"**{total_correct}**",
        f"**{overall_success:.2f}%**",
        f"**{overall_median_err:.4f}**",
        f"**{overall_max_err:.4f}**",
        f"**{overall_avg_time_ms:.1f}**"
    ])
    
    # Format and print the markdown table
    col_widths = [max(len(row[i]) for row in table_rows + [headers]) for i in range(len(headers))]
    
    def format_row(row):
        return "| " + " | ".join(row[i].ljust(col_widths[i]) for i in range(len(row))) + " |"
        
    print("\n" + format_row(headers))
    print("| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |")
    for r in table_rows:
        print(format_row(r))
        
    # Write to a file in artifacts for user review
    artifact_path = Path("/Users/suryodaypratapsingh/.gemini/antigravity-ide/brain/f02a8db8-6c65-460e-9136-a4ba3219f1e6/phase2_benchmark_report.md")
    with open(artifact_path, "w") as f:
        f.write("# Phase 2 Benchmark Evaluation Report\n\n")
        f.write("Evaluation of the final system against 200 procedurally generated pairs matching Phase 2 specifications:\n\n")
        f.write(format_row(headers) + "\n")
        f.write("| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |\n")
        for r in table_rows:
            f.write(format_row(r) + "\n")
            
    print(f"\nReport written to artifact: {artifact_path}")

if __name__ == "__main__":
    main()
