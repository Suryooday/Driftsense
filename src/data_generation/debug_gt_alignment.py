"""
Drift Sense - Quantitative Ground Truth Alignment Verification.

Crops the matching region from the search image, aligns/scales it,
and compares it pixel-by-pixel against the clean pre-transform reference image.
Computes Mean Absolute Difference (MAD) and peak cross-correlation offset.
"""

import os
import json
import glob
import random
import cv2
import numpy as np
from typing import Dict, Any


def run_quantitative_alignment_check(num_samples: int = 5) -> None:
    """
    Performs quantitative alignment checks on random samples and saves side-by-side debug images.
    """
    sample_dirs = sorted(glob.glob("data/sample_*"))
    if not sample_dirs:
        print("No sample directories found in data/.")
        return

    # Select 5 random samples
    random.seed(42)
    selected_dirs = random.sample(sample_dirs, min(len(sample_dirs), num_samples))

    print(f"=== Quantitative Ground Truth Alignment Check ===")
    print(f"{'Sample':<12} | {'MAD (0-255)':<12} | {'Peak Offset (Canvas px)':<25} | {'Status':<6}")
    print("-" * 65)

    os.makedirs("data/debug", exist_ok=True)

    for sample_dir in selected_dirs:
        sample_name = os.path.basename(sample_dir)
        
        # Load images and GT metadata
        search_img = cv2.imread(os.path.join(sample_dir, "search_image.png"), cv2.IMREAD_GRAYSCALE)
        ref_clean = cv2.imread(os.path.join(sample_dir, "reference_clean.png"), cv2.IMREAD_GRAYSCALE)
        
        with open(os.path.join(sample_dir, "ground_truth.json"), "r") as f:
            gt = json.load(f)

        true_x = gt["true_x"]
        true_y = gt["true_y"]
        zoom_ratio = gt["zoom_ratio"]
        ref_h, ref_w = ref_clean.shape

        # Re-scale search image to canvas scale to compare to clean reference
        search_canvas_w = int(search_img.shape[1] * zoom_ratio)
        search_canvas_h = int(search_img.shape[0] * zoom_ratio)
        search_canvas_img = cv2.resize(
            search_img, (search_canvas_w, search_canvas_h), interpolation=cv2.INTER_LINEAR
        )

        # Center coordinates on search canvas image
        rel_x_canvas = true_x * zoom_ratio
        rel_y_canvas = true_y * zoom_ratio

        # Crop the matching region from the search canvas image
        search_crop = cv2.getRectSubPix(
            search_canvas_img, (ref_w, ref_h), (rel_x_canvas, rel_y_canvas)
        )

        # 1. Compute Mean Absolute Difference (MAD)
        mad = np.mean(np.abs(search_crop.astype(float) - ref_clean.astype(float)))

        # 2. Compute cross-correlation peak offset
        # Crop center portion of ref_clean as template to avoid boundary effects in template matching
        tmpl_w, tmpl_h = ref_w // 2, ref_h // 2
        tmpl_x0 = (ref_w - tmpl_w) // 2
        tmpl_y0 = (ref_h - tmpl_h) // 2
        ref_tmpl = ref_clean[tmpl_y0 : tmpl_y0 + tmpl_h, tmpl_x0 : tmpl_x0 + tmpl_w]

        res = cv2.matchTemplate(search_crop, ref_tmpl, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        expected_x = tmpl_x0
        expected_y = tmpl_y0
        offset_x = max_loc[0] - expected_x
        offset_y = max_loc[1] - expected_y
        peak_offset = np.sqrt(offset_x**2 + offset_y**2)

        status = "PASS" if peak_offset <= 1.5 else "FAIL"

        print(f"{sample_name:<12} | {mad:<12.2f} | {peak_offset:<25.2f} | {status:<6}")

        # Save side-by-side verification image
        side_by_side = np.hstack((ref_clean, search_crop))
        # Draw text labels
        cv2.putText(side_by_side, "Ref Clean", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        cv2.putText(side_by_side, "Search Crop @ GT", (ref_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        
        cv2.imwrite(os.path.join("data/debug", f"{sample_name}_alignment_check.png"), side_by_side)

    print("-" * 65)
    print(f"Side-by-side alignment check images saved to: {os.path.abspath('data/debug')}\n")


if __name__ == "__main__":
    run_quantitative_alignment_check()
