"""
Drift-Sense Phase 2 Registration Entry Point (register.py).

Implements the official evaluation output contract:
    python register.py --input pairs.csv --output predictions.csv

Outputs predictions.csv with exact required columns:
    pair_id, x, y, theta, scale, found, score
"""

import os
import sys
import csv
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import cv2

# Add project root to python path to import modules
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.final_system import FinalSystemMatcher
from baseline import NaiveBaselineMatcher


def find_column(headers: List[str], candidates: List[str]) -> Optional[str]:
    headers_clean = [h.strip().strip("\ufeff\"'").lower() for h in headers]
    for cand in candidates:
        cand_clean = cand.lower()
        if cand_clean in headers_clean:
            idx = headers_clean.index(cand_clean)
            return headers[idx]
    return None


def resolve_image_path(p_str: str, input_dir: Path) -> str:
    """Robust path resolution relative to CSV directory, project root, and output folders."""
    p_clean = p_str.strip().strip("\"'").replace("\\", "/")
    candidates = [
        Path(p_clean),
        input_dir / p_clean,
        input_dir / p_clean.lstrip("/"),
        PROJECT_ROOT / p_clean,
        PROJECT_ROOT / p_clean.lstrip("/"),
        PROJECT_ROOT / "output" / p_clean,
        PROJECT_ROOT / "output" / p_clean.lstrip("/"),
        PROJECT_ROOT / "output_200" / p_clean,
        PROJECT_ROOT / "output_200" / p_clean.lstrip("/"),
        PROJECT_ROOT / "data" / p_clean,
        PROJECT_ROOT / "data" / p_clean.lstrip("/"),
        PROJECT_ROOT / "data" / Path(p_clean).name,
    ]
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return str(cand.resolve())
    return p_clean


def main():
    parser = argparse.ArgumentParser(description="Drift-Sense Wafer Alignment Registration")
    parser.add_argument("--input", required=True, help="Path to input CSV containing image pairs")
    parser.add_argument("--output", required=True, help="Path to write predictions CSV")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    input_dir = input_path.parent

    # Warm matchers for both Phase 1 and Phase 2 inputs
    phase1_matcher = FinalSystemMatcher()
    phase2_matcher = NaiveBaselineMatcher()

    print(f"Reading image pairs from: {args.input}")

    with open(input_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)

    # Detect column names (case-insensitive and robust to quotes/BOM)
    pair_col = find_column(headers, ["pair_id", "pairid", "id", "pair"])
    search_col = find_column(headers, ["search_path", "search_image_path", "search_image", "search", "wide_search"])
    ref_col = find_column(headers, ["reference_path", "reference_image_path", "reference_image", "reference", "ref_path", "ref"])

    if not all([pair_col, search_col, ref_col]):
        print("Warning: Could not match all standard headers. Falling back to default column index mapping (0, 1, 2).")
        p_idx, s_idx, r_idx = 0, 1, 2
    else:
        p_idx = headers.index(pair_col)
        s_idx = headers.index(search_col)
        r_idx = headers.index(ref_col)

    predictions = []
    total_pairs = len(rows)
    print(f"Processing {total_pairs} pairs with adaptive Phase 1/Phase 2 registration...")

    t_start = time.perf_counter()

    for i, row in enumerate(rows):
        if not row or len(row) <= max(p_idx, s_idx, r_idx):
            continue

        pair_id = row[p_idx].strip()
        srch_path_raw = row[s_idx]
        ref_path_raw = row[r_idx]

        srch_path = resolve_image_path(srch_path_raw, input_dir)
        ref_path = resolve_image_path(ref_path_raw, input_dir)

        t_pair_start = time.perf_counter()

        # Read images
        srch_img = cv2.imread(srch_path, cv2.IMREAD_GRAYSCALE)
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)

        if srch_img is None or ref_img is None:
            print(f"Error: Could not read images for pair {pair_id} ({srch_path_raw} or {ref_path_raw})")
            predictions.append({
                "pair_id": pair_id,
                "x": 0.0,
                "y": 0.0,
                "theta": 0.0,
                "scale": 0.0,
                "found": 0,
                "score": 0.0
            })
            continue

        ref_h, ref_w = ref_img.shape[:2]

        # Route matching based on reference dimensions:
        # Phase 2: Unscaled fine canvas (1000x1000, scale z in [8, 12])
        # Phase 1: Pre-scaled 256x256 crop
        if ref_h > 500 or ref_w > 500:
            res = phase2_matcher.match(ref_img, srch_img)
            pred_found = int(res["predicted_present"])
            pred_x = float(res["predicted_x"]) if pred_found else 0.0
            pred_y = float(res["predicted_y"]) if pred_found else 0.0
            pred_theta = float(res["predicted_theta"]) if pred_found else 0.0
            pred_scale = float(res["predicted_scale"]) if pred_found else 0.0
            pred_score = float(res["confidence_score"])
        else:
            p1_res = phase1_matcher.match(ref_img, srch_img)
            pred_found = int(p1_res.get("found", 1))
            pred_x = float(p1_res["predicted_x"]) if (pred_found and p1_res["predicted_x"] is not None) else 0.0
            pred_y = float(p1_res["predicted_y"]) if (pred_found and p1_res["predicted_y"] is not None) else 0.0
            pred_theta = float(p1_res["predicted_rotation"]) if pred_found else 0.0
            pred_scale = float(p1_res["predicted_scale"]) if pred_found else 0.0
            pred_score = float(p1_res["confidence_score"])

        predictions.append({
            "pair_id": pair_id,
            "x": round(pred_x, 4),
            "y": round(pred_y, 4),
            "theta": round(pred_theta, 4),
            "scale": round(pred_scale, 4),
            "found": pred_found,
            "score": round(pred_score, 4)
        })

        elapsed = time.perf_counter() - t_pair_start
        if (i + 1) % 5 == 0 or (i + 1) == total_pairs:
            print(f"[{i+1:03d}/{total_pairs:03d}] Pair {pair_id} | found={pred_found} | det=({pred_x:.1f}, {pred_y:.1f}) | th={pred_theta:.1f}° | z={pred_scale:.1f} | score={pred_score:.4f} | time={elapsed:.2f}s")

    t_total = time.perf_counter() - t_start
    print(f"\n✅ All {total_pairs} pairs processed in {t_total:.2f}s (Average: {t_total/total_pairs:.2f}s per pair)")

    # Write output predictions CSV strictly matching the contract
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["pair_id", "x", "y", "theta", "scale", "found", "score"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Predictions successfully written to: {args.output}")


if __name__ == "__main__":
    main()
