"""
Generate 5 multi-instance test datasets.
Each dataset contains a search image with the SAME reference pattern
placed at 3-5 different locations (one true target near center + decoys elsewhere).
Saves to data/multi_instance_test/ with ground truth CSV.
"""
import os
import sys
import csv
import json
import math
import numpy as np
import cv2
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from generate_dataset import (
    generate_wafer_canvas,
    extract_transformed_patch,
    apply_degradations
)

def paste_into(search_img: np.ndarray, patch: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """Paste a small patch centered at (cx, cy) into the search image."""
    ph, pw = patch.shape
    y0 = max(0, cy - ph // 2)
    x0 = max(0, cx - pw // 2)
    y1 = min(search_img.shape[0], y0 + ph)
    x1 = min(search_img.shape[1], x0 + pw)
    patch_crop = patch[:y1 - y0, :x1 - x0]
    search_img[y0:y1, x0:x1] = patch_crop
    return search_img


def main():
    output_dir = Path("data/multi_instance_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    rng = np.random.default_rng(7777)

    search_w, search_h = 1000, 1000
    ref_w, ref_h = 256, 256
    zoom_ratio = 10.0
    search_canvas_w = int(search_w * zoom_ratio)
    search_canvas_h = int(search_h * zoom_ratio)

    # 5 datasets, alternating DRAM / FinFET
    styles = ["DRAM", "FinFET", "DRAM", "FinFET", "DRAM"]

    # How many decoy copies to plant per dataset (besides the one true center copy)
    decoy_counts = [2, 3, 4, 2, 3]

    for i in range(5):
        style = styles[i]
        n_decoys = decoy_counts[i]
        density = rng.uniform(0.15, 0.35)

        canvas_w = search_canvas_w + 1500
        canvas_h = search_canvas_h + 1500

        canvas = generate_wafer_canvas(canvas_w, canvas_h, density, style, rng)

        # --- Search image: center crop at canvas midpoint ---
        search_cx = canvas_w / 2.0
        search_cy = canvas_h / 2.0

        search_img_clean = extract_transformed_patch(
            canvas, (search_cx, search_cy),
            (search_canvas_w, search_canvas_h), 0.0, 1.0
        )
        search_img_clean = cv2.resize(
            search_img_clean, (search_w, search_h), interpolation=cv2.INTER_AREA
        )

        # --- Reference: slight drift and rotation ---
        sample_rot   = rng.uniform(-3.5, 3.5)
        sample_scale = rng.uniform(0.88, 1.12)
        drift_x = rng.uniform(-4.0, 4.0)   # search-image pixels
        drift_y = rng.uniform(-4.0, 4.0)

        ref_cx_canvas = search_cx + drift_x * zoom_ratio
        ref_cy_canvas = search_cy + drift_y * zoom_ratio

        ref_img_clean = extract_transformed_patch(
            canvas, (ref_cx_canvas, ref_cy_canvas),
            (ref_w, ref_h), sample_rot, sample_scale
        )

        # Ground-truth position in search image
        tl_x_canvas = search_cx - search_canvas_w / 2.0
        tl_y_canvas = search_cy - search_canvas_h / 2.0
        true_x = (ref_cx_canvas - tl_x_canvas) / zoom_ratio
        true_y = (ref_cy_canvas - tl_y_canvas) / zoom_ratio

        # --- Down-scaled reference snippet to paste as decoys ---
        # The downscaled version represents what the reference looks like in search-image space
        ds_ref = cv2.resize(ref_img_clean, (ref_w // 10, ref_h // 10), interpolation=cv2.INTER_AREA)

        # Plant decoy copies CLOSE to center (60–200 px away)
        # This is a harder test: decoys are near the expected stage coordinate
        # but JUST outside the 50px drift zone the system crops to.
        decoy_positions = []
        min_dist = 60    # decoy must be at least this far from center (outside drift zone)
        max_dist = 200   # decoy must be at most this far from center (stays near center region)
        attempts = 0
        while len(decoy_positions) < n_decoys and attempts < 500:
            # Sample uniformly in the ring [min_dist, max_dist] around (500, 500)
            angle = rng.uniform(0, 2 * math.pi)
            dist  = rng.uniform(min_dist, max_dist)
            dx = int(500 + dist * math.cos(angle))
            dy = int(500 + dist * math.sin(angle))
            # Keep within image bounds
            if dx < 30 or dy < 30 or dx > search_w - 30 or dy > search_h - 30:
                attempts += 1
                continue
            # Ensure decoys don't overlap each other
            too_close = any(
                math.sqrt((dx - ex)**2 + (dy - ey)**2) < 60
                for ex, ey in decoy_positions
            )
            if not too_close:
                decoy_positions.append((dx, dy))
            attempts += 1

        # Build a clean search image, then paste decoys
        search_with_decoys = search_img_clean.copy()
        for (dcx, dcy) in decoy_positions:
            search_with_decoys = paste_into(search_with_decoys, ds_ref.copy(), dcx, dcy)

        # Apply SEM degradations
        search_img, _ = apply_degradations(
            search_with_decoys,
            noise_std=rng.uniform(0.02, 0.05),
            speckle_std=rng.uniform(0.01, 0.03),
            blur_sigma=rng.uniform(0.8, 1.5),
            charging_amp=float(rng.uniform(30.0, 70.0) * rng.choice([-1.0, 1.0])),
            rng=rng
        )
        ref_img, _ = apply_degradations(
            ref_img_clean,
            noise_std=rng.uniform(0.01, 0.03),
            speckle_std=rng.uniform(0.005, 0.015),
            blur_sigma=rng.uniform(0.5, 1.2),
            charging_amp=float(rng.uniform(10.0, 30.0) * rng.choice([-1.0, 1.0])),
            rng=rng
        )

        # --- Save files ---
        sample_dir = output_dir / f"sample_{i:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        srch_path = sample_dir / "search_image.png"
        ref_path  = sample_dir / "reference_image.png"

        cv2.imwrite(str(srch_path), search_img)
        cv2.imwrite(str(ref_path),  ref_img)

        # Annotated diagnostic: draw true position and decoy positions on search image
        diag = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)
        cv2.drawMarker(diag, (int(true_x), int(true_y)), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
        cv2.putText(diag, "TRUE", (int(true_x) + 8, int(true_y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        for j, (dcx, dcy) in enumerate(decoy_positions):
            cv2.drawMarker(diag, (dcx, dcy), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 2)
            cv2.putText(diag, f"DECOY {j+1}", (dcx + 8, dcy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.imwrite(str(sample_dir / "search_annotated.png"), diag)

        # Ground truth JSON
        gt = {
            "true_x": round(float(true_x), 4),
            "true_y": round(float(true_y), 4),
            "rotation_deg": round(float(sample_rot), 4),
            "scale_factor": round(float(sample_scale * zoom_ratio), 4),
            "zoom_ratio": float(zoom_ratio),
            "style": style,
            "num_decoys": n_decoys,
            "decoy_positions": [[dcx, dcy] for dcx, dcy in decoy_positions],
            "found": 1
        }
        with open(sample_dir / "ground_truth.json", "w") as f:
            json.dump(gt, f, indent=4)

        csv_rows.append({
            "pair_id": f"multi_instance_{i:03d}",
            "search_image_path": str(srch_path),
            "reference_image_path": str(ref_path),
            "GTx": round(float(true_x), 4),
            "GTy": round(float(true_y), 4),
            "GT_theta": round(float(sample_rot), 4),
            "GT_scale": round(float(sample_scale * zoom_ratio), 4),
            "GT_found": 1,
            "style": style,
            "num_decoys": n_decoys,
        })

        print(f"[{i+1}/5] Sample {i:03d} ({style}) | True=({true_x:.1f},{true_y:.1f}) | {n_decoys} decoys at {decoy_positions}")

    # Write CSV
    csv_path = output_dir / "ground_truth.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pair_id", "search_image_path", "reference_image_path",
            "GTx", "GTy", "GT_theta", "GT_scale", "GT_found", "style", "num_decoys"
        ])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nDataset saved to: {output_dir.resolve()}")
    print(f"CSV saved to:     {csv_path.resolve()}")
    print("\nEach sample folder contains:")
    print("  search_image.png      - noisy search image with decoys embedded")
    print("  reference_image.png   - reference template (rotated/scaled)")
    print("  search_annotated.png  - annotated diagnostic (green=TRUE, red=DECOY)")
    print("  ground_truth.json     - true coords, rotation, scale, decoy positions")


if __name__ == "__main__":
    main()
