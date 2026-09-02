"""
Drift-Sense Phase 2 — Visual QA Contact Sheet Module (contact_sheet.py).

Renders a consolidated visual inspection contact sheet (contact_sheet.png) displaying
all image pairs with ground-truth instance footprints, reference insets, and presence badges.
"""

import os
import sys
import csv
import math
import argparse
from typing import Dict, List, Any
import numpy as np
import cv2

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.geometry import rotation_matrix


def draw_oriented_bbox(
    img: np.ndarray,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    theta_deg: float,
    color: tuple = (0, 255, 0),
    thickness: int = 2
):
    """Draws rotated bounding box representing the true reference footprint in search image."""
    R = rotation_matrix(theta_deg)
    hw, hh = width / 2.0, height / 2.0
    corners = np.array([
        [-hw, -hh],
        [ hw, -hh],
        [ hw,  hh],
        [-hw,  hh]
    ], dtype=np.float64)

    # Rotate and translate
    rot_corners = corners @ R.T + np.array([center_x, center_y])
    pts = rot_corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
    # Center crosshair
    cx, cy = int(round(center_x)), int(round(center_y))
    cv2.drawMarker(img, (cx, cy), color, markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)


def render_contact_sheet(
    data_dir: str = "output",
    output_path: str = None,
    max_cols: int = 4,
    thumb_size: int = 400
) -> str:
    if output_path is None:
        output_path = os.path.join(data_dir, "contact_sheet.png")

    gt_csv_path = os.path.join(data_dir, "ground_truth.csv")
    pairs_csv_path = os.path.join(data_dir, "pairs.csv")
    manifest_csv_path = os.path.join(data_dir, "manifest.csv")

    if not os.path.exists(gt_csv_path) or not os.path.exists(pairs_csv_path):
        raise FileNotFoundError(f"Missing ground_truth.csv or pairs.csv in {data_dir}")

    gt_dict = {}
    with open(gt_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            gt_dict[r["pair_id"]] = {
                "present": int(r["present"]),
                "x": float(r["x"]),
                "y": float(r["y"]),
                "theta": float(r["theta"]),
                "scale": float(r["scale"])
            }

    manifest_dict = {}
    if os.path.exists(manifest_csv_path):
        with open(manifest_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                manifest_dict[r["pair_id"]] = r

    pairs = []
    with open(pairs_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pairs.append(r)

    # For contact sheet, limit to first 20 pairs if large
    pairs_to_draw = pairs[:20]
    num_items = len(pairs_to_draw)
    cols = min(num_items, max_cols)
    rows = int(math.ceil(num_items / float(cols)))

    cell_w, cell_h = thumb_size, thumb_size + 60
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 30, dtype=np.uint8)

    for idx, p in enumerate(pairs_to_draw):
        pid = p["pair_id"]
        srch_p = p["search_path"]
        ref_p = p["reference_path"]
        if not os.path.isabs(srch_p):
            srch_p = os.path.join(data_dir, srch_p)
        if not os.path.isabs(ref_p):
            ref_p = os.path.join(data_dir, ref_p)

        gt = gt_dict[pid]
        meta = manifest_dict.get(pid, {})
        set_name = meta.get("set", "A" if gt["present"] == 1 else "C")
        preset = meta.get("preset", "semicon")
        sev = meta.get("severity_level", "0")

        srch_img = cv2.imread(srch_p, cv2.IMREAD_UNCHANGED)
        ref_img = cv2.imread(ref_p, cv2.IMREAD_UNCHANGED)

        if len(srch_img.shape) == 2:
            srch_bgr = cv2.cvtColor(srch_img, cv2.COLOR_GRAY2BGR)
        else:
            srch_bgr = srch_img.copy()

        if len(ref_img.shape) == 2:
            ref_bgr = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
        else:
            ref_bgr = ref_img.copy()

        # Draw annotations on full-size search image first
        if gt["present"] == 1:
            z = gt["scale"]
            th = gt["theta"]
            box_w = 1000.0 / z
            box_h = 1000.0 / z
            draw_oriented_bbox(srch_bgr, gt["x"], gt["y"], box_w, box_h, th, color=(0, 255, 0), thickness=3)
        else:
            # Mark absent
            cv2.line(srch_bgr, (50, 50), (950, 950), (0, 0, 220), 4)
            cv2.line(srch_bgr, (50, 950), (950, 50), (0, 0, 220), 4)
            cv2.putText(srch_bgr, "ABSENT", (300, 520), cv2.FONT_HERSHEY_DUPLEX, 2.2, (0, 0, 255), 4)

        # Scale search image to thumbnail
        srch_thumb = cv2.resize(srch_bgr, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)

        # Inset reference image in top-right corner
        inset_size = int(thumb_size * 0.28)
        ref_thumb = cv2.resize(ref_bgr, (inset_size, inset_size), interpolation=cv2.INTER_AREA)
        # Border around inset
        cv2.rectangle(ref_thumb, (0, 0), (inset_size - 1, inset_size - 1), (255, 255, 255), 2)
        srch_thumb[8:8 + inset_size, thumb_size - inset_size - 8:thumb_size - 8] = ref_thumb

        # Compose cell with header / text box below
        cell = np.full((cell_h, cell_w, 3), 20, dtype=np.uint8)
        cell[:thumb_size, :] = srch_thumb

        # Text banner
        r_idx = idx // cols
        c_idx = idx % cols

        title_col = (0, 255, 255) if gt["present"] == 1 else (0, 100, 255)
        txt1 = f"{pid} | Set {set_name} ({preset})"
        txt2 = f"z={gt['scale']:.2f} | th={gt['theta']:.2f} deg | Sev={sev}" if gt["present"] == 1 else "ABSENT (present=0)"

        cv2.putText(cell, txt1, (10, thumb_size + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, title_col, 1)
        cv2.putText(cell, txt2, (10, thumb_size + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

        sheet[r_idx * cell_h:(r_idx + 1) * cell_h, c_idx * cell_w:(c_idx + 1) * cell_w] = cell

    cv2.imwrite(output_path, sheet)
    print(f"Contact sheet successfully rendered to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drift-Sense Phase 2 Contact Sheet Visual QA")
    parser.add_argument("--data-dir", default="output", help="Directory containing dataset and CSVs")
    parser.add_argument("--output", default="output/contact_sheet.png", help="Path to output image")
    args = parser.parse_args()

    render_contact_sheet(args.data_dir, args.output)
