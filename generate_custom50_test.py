"""
Generate 50 test pairs (Phase 2-style) with:
  - Reference image extracted from a position offset between 100–400 px from center
    (NOT a corner, NOT at center — intermediate range, harder for gated crop)
  - Type distribution:
      0–14  : Greyscale Nominal (DRAM/FinFET)     — Set A
      15–29 : Greyscale Degraded (DRAM/FinFET)    — Set B
      30–39 : Greyscale Absent (Set C)             — Set C
      40–49 : RGB Optical analogue (Set D)         — Set D
  - Zoom ratio 10x, rotation ±5°, final scale [8,12]×
  - Saved to data/custom50_test/
"""
import os, sys, csv, json
import numpy as np
import cv2
from pathlib import Path

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from generate_dataset import (
    generate_wafer_canvas,
    extract_transformed_patch,
    apply_degradations,
)


# ──────────────────────────────────────────────────────────────────────────────
# Optical coloriser  (warm-tinted BGR, mimics optical microscope)
# ──────────────────────────────────────────────────────────────────────────────
def colorize_optical(gray: np.ndarray) -> np.ndarray:
    """Convert uint8 greyscale → warm-tinted BGR optical analogue."""
    f = gray.astype(np.float32) / 255.0
    h, w = gray.shape
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    bgr[..., 0] = np.clip(60  + f * (30  - 60),  0, 255).astype(np.uint8)  # B
    bgr[..., 1] = np.clip(25  + f * (190 - 25),  0, 255).astype(np.uint8)  # G
    bgr[..., 2] = np.clip(40  + f * (250 - 40),  0, 255).astype(np.uint8)  # R
    return bgr


def add_rgb_sem_noise(rgb: np.ndarray, rng: np.random.Generator,
                      noise_std: float = 0.015, blur_sigma: float = 1.0) -> np.ndarray:
    """Light noise + blur for the RGB optical channel."""
    out = rgb.astype(np.float32) / 255.0
    out += rng.normal(0, noise_std, out.shape).astype(np.float32)
    out = np.clip(out, 0, 1)
    out_uint8 = (out * 255).astype(np.uint8)
    for c in range(3):
        out_uint8[..., c] = cv2.GaussianBlur(out_uint8[..., c], (0, 0), blur_sigma)
    return out_uint8


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    output_dir = Path("data/custom50_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42_000)

    zoom_ratio      = 10.0
    search_w, search_h = 1000, 1000
    ref_w,  ref_h   = 256, 256
    rot_min, rot_max = -5.0, 5.0
    scale_min, scale_max = 0.8, 1.2   # → final scale in [8, 12]×

    search_canvas_w = int(search_w * zoom_ratio)
    search_canvas_h = int(search_h * zoom_ratio)
    canvas_w = search_canvas_w + 1500
    canvas_h = search_canvas_h + 1500

    csv_rows = []

    print("Generating 50 custom test pairs ...")
    print("  0-14  -> Set A  Nominal  (greyscale)")
    print("  15-29 -> Set B  Degraded (greyscale)")
    print("  30-39 -> Set C  Absent   (greyscale)")
    print("  40-49 -> Set D  Optical  (RGB 3-channel)\n")

    for i in range(50):
        # ── Set & style ───────────────────────────────────────────────────────
        if   i < 15: set_type = "A"
        elif i < 30: set_type = "B"
        elif i < 40: set_type = "C"
        else:        set_type = "D"

        style = "DRAM" if (i % 2 == 0) else "FinFET"

        # ── Degradation params ────────────────────────────────────────────────
        density = rng.uniform(0.12, 0.38)

        if set_type == "A":
            noise   = rng.uniform(0.01,  0.03)
            speckle = rng.uniform(0.005, 0.02)
            blur    = rng.uniform(0.5,   1.2)
            charge  = float(rng.uniform(10, 35)  * rng.choice([-1., 1.]))
        elif set_type == "B":
            noise   = rng.uniform(0.04,  0.09)
            speckle = rng.uniform(0.025, 0.07)
            blur    = rng.uniform(1.2,   2.8)
            charge  = float(rng.uniform(50, 100) * rng.choice([-1., 1.]))
        elif set_type == "C":
            noise   = rng.uniform(0.01,  0.04)
            speckle = rng.uniform(0.005, 0.03)
            blur    = rng.uniform(0.5,   1.5)
            charge  = float(rng.uniform(15, 55)  * rng.choice([-1., 1.]))
        else:   # D optical
            noise   = rng.uniform(0.008, 0.02)
            speckle = rng.uniform(0.003, 0.012)
            blur    = rng.uniform(0.8,   1.6)
            charge  = 0.0

        sample_rot   = rng.uniform(rot_min, rot_max)
        sample_scale = rng.uniform(scale_min, scale_max)

        # ── Build search canvas ───────────────────────────────────────────────
        canvas_search = generate_wafer_canvas(canvas_w, canvas_h, density, style, rng)

        search_cx = rng.uniform(search_canvas_w / 2.0, canvas_w - search_canvas_w / 2.0)
        search_cy = rng.uniform(search_canvas_h / 2.0, canvas_h - search_canvas_h / 2.0)

        search_clean_canvas = extract_transformed_patch(
            canvas_search, (search_cx, search_cy),
            (search_canvas_w, search_canvas_h), 0.0, 1.0
        )
        search_clean = cv2.resize(search_clean_canvas, (search_w, search_h),
                                  interpolation=cv2.INTER_AREA)

        # ── Reference position ────────────────────────────────────────────────
        # KEY: offset the reference 100-400 px from center in search-image space
        is_absent = (set_type == "C")
        offset_dist = None

        if is_absent:
            ref_density = rng.uniform(0.12, 0.38)
            canvas_ref  = generate_wafer_canvas(canvas_w, canvas_h, ref_density, style, rng)
            ref_cx = rng.uniform(ref_w * 5, canvas_w - ref_w * 5)
            ref_cy = rng.uniform(ref_h * 5, canvas_h - ref_h * 5)
            true_x = true_y = 0.0
            scale_factor = rotation_deg = 0.0
            found = 0
        else:
            canvas_ref = canvas_search

            # Offset 100-400 px in search-image space
            offset_dist  = rng.uniform(100.0, 400.0)
            offset_angle = rng.uniform(0, 2 * np.pi)
            offset_x_search = offset_dist * np.cos(offset_angle)
            offset_y_search = offset_dist * np.sin(offset_angle)

            ref_cx = search_cx + offset_x_search * zoom_ratio
            ref_cy = search_cy + offset_y_search * zoom_ratio

            # Clamp within canvas bounds
            margin = max(ref_w, ref_h) * zoom_ratio * 0.7
            ref_cx = float(np.clip(ref_cx, margin, canvas_w - margin))
            ref_cy = float(np.clip(ref_cy, margin, canvas_h - margin))

            tl_x   = search_cx - search_canvas_w / 2.0
            tl_y   = search_cy - search_canvas_h / 2.0
            true_x = (ref_cx - tl_x) / zoom_ratio
            true_y = (ref_cy - tl_y) / zoom_ratio

            scale_factor = sample_scale * zoom_ratio
            rotation_deg = sample_rot
            found        = 1

        # ── Extract reference patch ───────────────────────────────────────────
        ref_clean = extract_transformed_patch(
            canvas_ref, (ref_cx, ref_cy),
            (ref_w, ref_h), sample_rot, sample_scale
        )

        # ── Apply degradations ────────────────────────────────────────────────
        search_deg, _ = apply_degradations(
            search_clean, noise_std=noise, speckle_std=speckle,
            blur_sigma=blur, charging_amp=charge, rng=rng
        )
        ref_deg, _ = apply_degradations(
            ref_clean, noise_std=noise, speckle_std=speckle,
            blur_sigma=blur, charging_amp=charge * 0.5, rng=rng
        )

        # ── Colourise Set D ───────────────────────────────────────────────────
        if set_type == "D":
            search_out = add_rgb_sem_noise(
                colorize_optical(search_deg), rng,
                noise_std=float(rng.uniform(0.008, 0.018)),
                blur_sigma=float(rng.uniform(0.5, 1.0))
            )
            ref_out = add_rgb_sem_noise(
                colorize_optical(ref_deg), rng,
                noise_std=float(rng.uniform(0.005, 0.012)),
                blur_sigma=float(rng.uniform(0.4, 0.8))
            )
        else:
            search_out = search_deg
            ref_out    = ref_deg

        # ── Save ──────────────────────────────────────────────────────────────
        sample_dir = output_dir / f"sample_{i:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        srch_path = sample_dir / "search_image.png"
        ref_path  = sample_dir / "reference_image.png"

        cv2.imwrite(str(srch_path), search_out)
        cv2.imwrite(str(ref_path),  ref_out)

        # Annotated diagnostic
        if set_type == "D":
            diag = search_out.copy()
        else:
            diag = cv2.cvtColor(search_out, cv2.COLOR_GRAY2BGR)

        if found:
            cv2.drawMarker(diag, (int(true_x), int(true_y)),
                           (0, 255, 0), cv2.MARKER_CROSS, 40, 2)
            cv2.putText(diag, f"GT ({true_x:.0f},{true_y:.0f})",
                        (int(true_x) + 6, int(true_y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(str(sample_dir / "search_annotated.png"), diag)

        # Ground-truth JSON
        gt_json = {
            "true_x":                       round(float(true_x),        4),
            "true_y":                       round(float(true_y),        4),
            "rotation_deg":                 round(float(rotation_deg),  4),
            "scale_factor":                 round(float(scale_factor),  4),
            "zoom_ratio":                   float(zoom_ratio),
            "noise_level":                  float(noise),
            "style":                        style,
            "set":                          set_type,
            "found":                        int(found),
            "ref_offset_from_center_px":    round(float(offset_dist), 2) if offset_dist else None,
            "image_channels":               3 if set_type == "D" else 1,
        }
        with open(sample_dir / "ground_truth.json", "w") as f:
            json.dump(gt_json, f, indent=4)

        csv_rows.append({
            "pair_id":              f"custom50_{i:03d}",
            "search_image_path":    str(srch_path),
            "reference_image_path": str(ref_path),
            "GTx":                  round(float(true_x),       4),
            "GTy":                  round(float(true_y),       4),
            "GT_theta":             round(float(rotation_deg), 4),
            "GT_scale":             round(float(scale_factor), 4),
            "GT_found":             int(found),
            "set":                  set_type,
            "style":                style,
            "channels":             3 if set_type == "D" else 1,
        })

        dist_str = f"offset={offset_dist:.0f}px" if offset_dist else "absent"
        print(f"[{i+1:>2}/50] {f'sample_{i:03d}'} | Set {set_type} | {style:<6} | "
              f"{dist_str} | GT=({true_x:.1f},{true_y:.1f})")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    csv_path = output_dir / "ground_truth.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pair_id", "search_image_path", "reference_image_path",
            "GTx", "GTy", "GT_theta", "GT_scale", "GT_found",
            "set", "style", "channels"
        ])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nDataset saved to: {output_dir.resolve()}")
    print(f"CSV saved to:     {csv_path.resolve()}")
    print("\nEach sample folder contains:")
    print("  search_image.png      — 1000x1000  greyscale (A/B/C) or RGB (D)")
    print("  reference_image.png   — 256x256    greyscale or RGB")
    print("  search_annotated.png  — GT marker overlay")
    print("  ground_truth.json     — coords, rotation, scale, offset, channels")


if __name__ == "__main__":
    main()
