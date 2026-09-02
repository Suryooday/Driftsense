import os
import sys
import csv
import argparse
import time
import cv2
from pathlib import Path

# Add project root to python path to import modules
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.final_system import FinalSystemMatcher

def find_column(headers, candidates):
    headers_clean = [h.strip().strip("\ufeff\"'").lower() for h in headers]
    for cand in candidates:
        cand_clean = cand.lower()
        if cand_clean in headers_clean:
            idx = headers_clean.index(cand_clean)
            return headers[idx]
    return None

def resolve_image_path(p_str):
    p_clean = p_str.strip().strip("\"'").replace("\\", "/")
    candidates = [
        Path(p_clean),
        PROJECT_ROOT / p_clean,
        PROJECT_ROOT / p_clean.lstrip("/"),
        PROJECT_ROOT / "data" / p_clean,
        PROJECT_ROOT / "data" / p_clean.lstrip("/"),
        PROJECT_ROOT / "data" / Path(p_clean).name,
    ]
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return str(cand.resolve())
    return p_clean

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Register Wafer Alignment")
    parser.add_argument("--input", required=True, help="Path to input CSV containing image pairs")
    parser.add_argument("--output", required=True, help="Path to write predictions CSV")
    args = parser.parse_args()
    
    matcher = FinalSystemMatcher()
    
    print(f"Reading image pairs from: {args.input}")
    
    with open(args.input, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)
        
    # Detect column names (case-insensitive and robust to quotes/BOM)
    pair_col = find_column(headers, ["pair_id", "pairid", "id"])
    search_col = find_column(headers, ["search_image_path", "search_image", "search", "search_path"])
    ref_col = find_column(headers, ["reference_image_path", "reference_image", "reference", "ref_path", "ref"])
    
    if not all([pair_col, search_col, ref_col]):
        # Fall back to index-based if header matching fails
        print("Warning: Could not match all headers. Falling back to column index mapping.")
        p_idx, s_idx, r_idx = 0, 1, 2
    else:
        p_idx = headers.index(pair_col)
        s_idx = headers.index(search_col)
        r_idx = headers.index(ref_col)
        
    predictions = []
    
    total_pairs = len(rows)
    print(f"Processing {total_pairs} pairs...")
    
    t_start = time.perf_counter()
    
    for i, row in enumerate(rows):
        if not row or len(row) <= max(p_idx, s_idx, r_idx):
            continue
            
        pair_id = row[p_idx].strip()
        srch_path_raw = row[s_idx]
        ref_path_raw = row[r_idx]
        
        srch_path = resolve_image_path(srch_path_raw)
        ref_path = resolve_image_path(ref_path_raw)
        
        t_pair_start = time.perf_counter()
        
        # Read images
        srch_img = cv2.imread(srch_path, cv2.IMREAD_GRAYSCALE)
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        
        if srch_img is None or ref_img is None:
            print(f"Error: Could not read images for pair {pair_id} ({srch_path_raw} or {ref_path_raw})")
            # Write default absent
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
            
        # Match
        res = matcher.match(ref_img, srch_img)
        
        predictions.append({
                "pair_id": pair_id,
                "x":     res["predicted_x"]      if res["found"]==1 else 0.0,
                "y":     res["predicted_y"]      if res["found"]==1 else 0.0,
                "theta": res["predicted_rotation"] if res["found"]==1 else 0.0,
                "scale": res["predicted_scale"]  if res["found"]==1 else 0.0,
                "found": res["found"],
                "score": res["confidence_score"],
                "ncc_score":  res.get("ncc_score",  res["confidence_score"]),
                "ssim_score": res.get("ssim_score", 0.0),
            })
            
        elapsed = time.perf_counter() - t_pair_start
        print(f"[{i+1}/{total_pairs}] Pair {pair_id} | found={predictions[-1]['found']} | score={predictions[-1]['score']:.4f} | time={elapsed:.2f}s")
        
    t_total = time.perf_counter() - t_start
    print(f"All pairs processed in {t_total:.2f}s (Average: {t_total/total_pairs:.2f}s per pair)")
    
    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "x", "y", "theta", "scale", "found", "score", "ncc_score", "ssim_score"])
        writer.writeheader()
        writer.writerows(predictions)
        
    print(f"Predictions successfully written to: {args.output}")

if __name__ == "__main__":
    main()
