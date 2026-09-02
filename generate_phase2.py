"""
Drift-Sense Phase 2 — Main Dataset Generator Entry Point (generate_phase2.py).

Emits reference/search pairs with unknown zoom z in [8.0, 12.0], unknown rotation theta in [-5.0, 5.0] deg,
and reference-absent pairs with provably verifiable ground-truth labels.

CLI Arguments:
  --output-dir : Target directory (default: output)
  --seed       : Random seed for deterministic generation (default: 2026)
  --pairs      : Number of pairs to generate (default: 20, supports up to 200)
"""

import os
import sys
import csv
import math
import argparse
import time
from typing import Dict, List, Tuple, Any
import numpy as np
import cv2

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.presets import get_preset, PRESETS
from src.patterns.zones import generate_zone_canvas
from src.sem_imaging import (
    gaussian_psf_blur,
    add_shot_noise,
    add_detector_noise,
    add_speckle_noise,
    add_salt_and_pepper_noise,
    add_charging_streaks,
    apply_raster_drift,
    apply_vignette,
    apply_gamma,
    apply_barrel_distortion
)
from src.geometry import (
    get_canvas_to_search_matrix,
    get_search_to_canvas_matrix,
    transform_point,
    calculate_required_canvas_size
)
from src.resampling import warp_affine_antialiased
from src.optical import simulate_optical_image
from src.verifier import verify_disk_pair


# Hand-specified deterministic table for the 20-pair core benchmark (Section 2.4)
CORE_20_PAIRS_SPEC = [
    # Set A — Nominal (8 pairs: 4 distinctive on-grid/edge, 4 intra-mat periodic)
    {"pair_id": "p000", "set": "A", "preset": "dram_1x",       "z": 8.00,  "theta": -4.90, "present": 1, "sev": 0, "ambiguous": False},
    {"pair_id": "p001", "set": "A", "preset": "finfet_7nm",     "z": 12.00, "theta":  4.90, "present": 1, "sev": 0, "ambiguous": True},
    {"pair_id": "p002", "set": "A", "preset": "dram_dense",     "z": 10.00, "theta":  0.00, "present": 1, "sev": 0, "ambiguous": False},
    {"pair_id": "p003", "set": "A", "preset": "finfet_14nm",    "z": 8.50,  "theta":  2.50, "present": 1, "sev": 0, "ambiguous": True},
    {"pair_id": "p004", "set": "A", "preset": "dram_loose",     "z": 11.20, "theta": -2.80, "present": 1, "sev": 0, "ambiguous": False},
    {"pair_id": "p005", "set": "A", "preset": "finfet_22nm",    "z": 9.40,  "theta":  1.20, "present": 1, "sev": 0, "ambiguous": True},
    {"pair_id": "p006", "set": "A", "preset": "dram_wide",      "z": 10.80, "theta": -3.50, "present": 1, "sev": 0, "ambiguous": False},
    {"pair_id": "p007", "set": "A", "preset": "finfet_45nm",    "z": 9.00,  "theta": -1.50, "present": 1, "sev": 0, "ambiguous": True},

    # Set B — Degraded (6 pairs across 4 monotonic severity levels)
    {"pair_id": "p008", "set": "B", "preset": "finfet_10nm",    "z": 10.50, "theta":  1.80, "present": 1, "sev": 1, "ambiguous": False},
    {"pair_id": "p009", "set": "B", "preset": "dram_compact",   "z": 9.80,  "theta": -2.20, "present": 1, "sev": 2, "ambiguous": False},
    {"pair_id": "p010", "set": "B", "preset": "finfet_28nm",    "z": 11.50, "theta":  3.10, "present": 1, "sev": 2, "ambiguous": True},
    {"pair_id": "p011", "set": "B", "preset": "dram_legacy",    "z": 8.80,  "theta": -0.80, "present": 1, "sev": 3, "ambiguous": False},
    {"pair_id": "p012", "set": "B", "preset": "finfet_7nm",     "z": 10.00, "theta":  0.00, "present": 1, "sev": 3, "ambiguous": False}, # Grid-aligned -> high sev
    {"pair_id": "p013", "set": "B", "preset": "dram_1x",        "z": 9.00,  "theta":  0.00, "present": 1, "sev": 4, "ambiguous": False}, # Grid-aligned -> max sev

    # Set C — Absent (4 pairs, present=0, same-family macro-decoys)
    {"pair_id": "p014", "set": "C", "preset": "dram_dense",     "z": 9.50,  "theta":  2.00, "present": 0, "sev": 0, "ambiguous": False},
    {"pair_id": "p015", "set": "C", "preset": "finfet_14nm",    "z": 11.00, "theta": -3.00, "present": 0, "sev": 0, "ambiguous": False},
    {"pair_id": "p016", "set": "C", "preset": "dram_loose",     "z": 8.50,  "theta":  1.50, "present": 0, "sev": 0, "ambiguous": False},
    {"pair_id": "p017", "set": "C", "preset": "finfet_22nm",    "z": 10.20, "theta": -1.00, "present": 0, "sev": 0, "ambiguous": False},

    # Set D — Optical Microscope Analogue (2 pairs, 3-channel RGB)
    {"pair_id": "p018", "set": "D", "preset": "dram_wide",      "z": 9.20,  "theta":  1.40, "present": 1, "sev": 0, "ambiguous": False},
    {"pair_id": "p019", "set": "D", "preset": "finfet_28nm",    "z": 10.60, "theta": -2.40, "present": 1, "sev": 0, "ambiguous": True},
]


def build_pair_specifications(num_pairs: int, rng: np.random.Generator) -> List[Dict[str, Any]]:
    if num_pairs == 20:
        return list(CORE_20_PAIRS_SPEC)

    preset_names = list(PRESETS.keys())
    n_a = int(round(num_pairs * 0.40))
    n_b = int(round(num_pairs * 0.30))
    n_c = int(round(num_pairs * 0.20))
    n_d = num_pairs - (n_a + n_b + n_c)

    specs = []
    pid_idx = 0

    # Set A (Nominal)
    for i in range(n_a):
        preset = preset_names[pid_idx % len(preset_names)]
        if i == 0:
            z, th = 8.00, -4.90
        elif i == 1:
            z, th = 12.00, 4.90
        elif i == 2:
            z, th = 10.00, 0.00
        else:
            z = float(rng.uniform(8.0, 12.0))
            th = float(rng.uniform(-5.0, 5.0))
        amb = (i % 2 == 1)
        specs.append({
            "pair_id": f"p{pid_idx:03d}",
            "set": "A",
            "preset": preset,
            "z": round(z, 2),
            "theta": round(th, 2),
            "present": 1,
            "sev": 0,
            "ambiguous": amb
        })
        pid_idx += 1

    # Set B (Degraded, 4 monotonic severity levels)
    for i in range(n_b):
        preset = preset_names[pid_idx % len(preset_names)]
        sev = (i % 4) + 1  # 1, 2, 3, 4
        if sev >= 3:
            z = float(rng.choice([8.0, 9.0, 10.0, 11.0, 12.0]))
            th = float(rng.choice([-4.0, -2.0, 0.0, 2.0, 4.0]))
        else:
            z = float(rng.uniform(8.2, 11.8))
            th = float(rng.uniform(-4.8, 4.8))
        specs.append({
            "pair_id": f"p{pid_idx:03d}",
            "set": "B",
            "preset": preset,
            "z": round(z, 2),
            "theta": round(th, 2),
            "present": 1,
            "sev": sev,
            "ambiguous": (sev == 2 and i % 2 == 1)
        })
        pid_idx += 1

    # Set C (Absent)
    for i in range(n_c):
        preset = preset_names[pid_idx % len(preset_names)]
        z = float(rng.uniform(8.0, 12.0))
        th = float(rng.uniform(-5.0, 5.0))
        specs.append({
            "pair_id": f"p{pid_idx:03d}",
            "set": "C",
            "preset": preset,
            "z": round(z, 2),
            "theta": round(th, 2),
            "present": 0,
            "sev": 0,
            "ambiguous": False
        })
        pid_idx += 1

    # Set D (Optical Analogue)
    for i in range(n_d):
        preset = preset_names[pid_idx % len(preset_names)]
        z = float(rng.uniform(8.0, 12.0))
        th = float(rng.uniform(-5.0, 5.0))
        specs.append({
            "pair_id": f"p{pid_idx:03d}",
            "set": "D",
            "preset": preset,
            "z": round(z, 2),
            "theta": round(th, 2),
            "present": 1,
            "sev": 0,
            "ambiguous": (i % 2 == 1)
        })
        pid_idx += 1

    return specs


def apply_severity_degradations(
    img: np.ndarray,
    severity: int,
    rng: np.random.Generator
) -> Tuple[np.ndarray, float]:
    """
    Applies calibrated SEM degradations strictly following the monotonic difficulty ladder.
    """
    if severity == 0:
        # Nominal
        out = gaussian_psf_blur(img, spot_size_nm=5.0, pixel_size_nm=10.0)
        out = add_shot_noise(out, dose=2000.0, rng=rng)
        out = add_detector_noise(out, sigma=2.0, rng=rng)
        return out, 0.0

    elif severity == 1:
        # Severity 1 (Mild noise & small spot blur, sub-pixel error ~ 0.3 px)
        out = gaussian_psf_blur(img, spot_size_nm=7.0, pixel_size_nm=10.0, astigmatism_ratio=1.12)
        out = add_shot_noise(out, dose=850.0, rng=rng)
        out = add_detector_noise(out, sigma=5.0, rng=rng)
        out = add_speckle_noise(out, sigma=0.03, rng=rng)
        return out, 0.0

    elif severity == 2:
        # Severity 2 (Moderate noise, spot blur ~ 12.5nm, error ~ 1.5 - 3.5 px, credit ~ 0.4 - 0.6)
        out = gaussian_psf_blur(img, spot_size_nm=12.5, pixel_size_nm=10.0, astigmatism_ratio=1.28)
        out = add_shot_noise(out, dose=220.0, rng=rng)
        out = add_detector_noise(out, sigma=15.0, rng=rng)
        out = add_speckle_noise(out, sigma=0.09, rng=rng)
        out = add_charging_streaks(out, streak_prob=0.8, intensity=0.9, rng=rng)
        out = apply_vignette(out, strength=0.25)
        return out, 0.0

    elif severity == 3:
        # Severity 3 (High noise, spot blur ~ 18nm, pushes baseline peak near/below threshold 0.55)
        out = gaussian_psf_blur(img, spot_size_nm=18.0, pixel_size_nm=10.0, astigmatism_ratio=1.42)
        out = add_shot_noise(out, dose=90.0, rng=rng)
        out = add_detector_noise(out, sigma=26.0, rng=rng)
        out = add_speckle_noise(out, sigma=0.14, rng=rng)
        out = add_charging_streaks(out, streak_prob=1.8, intensity=1.8, rng=rng)
        out = apply_vignette(out, strength=0.38)
        out = apply_gamma(out, gamma=1.35)
        return out, 0.0

    else:  # severity == 4 (Maximum degradation: heavy shot & detector noise, large blur ~ 25nm)
        out = gaussian_psf_blur(img, spot_size_nm=25.0, pixel_size_nm=10.0, astigmatism_ratio=1.6)
        out = add_shot_noise(out, dose=40.0, rng=rng)
        out = add_detector_noise(out, sigma=38.0, rng=rng)
        out = add_speckle_noise(out, sigma=0.22, rng=rng)
        out = add_charging_streaks(out, streak_prob=3.2, intensity=2.5, rng=rng)
        out = apply_vignette(out, strength=0.50)
        out = apply_gamma(out, gamma=1.60)
        out = add_salt_and_pepper_noise(out, prob=0.02, rng=rng)
        return out, 0.0


def generate_single_pair(
    spec: Dict[str, Any],
    output_dir: str,
    rng: np.random.Generator,
    max_retries: int = 30
) -> Dict[str, Any]:
    pid = spec["pair_id"]
    preset_name = spec["preset"]
    preset = get_preset(preset_name)
    kind = preset["kind"]
    z = spec["z"]
    theta = spec["theta"]
    present = spec["present"]
    set_type = spec["set"]
    severity = spec["sev"]
    is_ambiguous = spec.get("ambiguous", False)

    search_w, search_h = 1000, 1000
    ref_w, ref_h = 1000, 1000
    c_search = ((search_w - 1) / 2.0, (search_h - 1) / 2.0)

    canvas_dim = calculate_required_canvas_size(search_w, search_h, z_max=z, margin_px=2000)
    c_canvas = ((canvas_dim - 1) / 2.0, (canvas_dim - 1) / 2.0)

    ref_dir = os.path.join(output_dir, "reference")
    srch_dir = os.path.join(output_dir, "search")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(srch_dir, exist_ok=True)

    ref_path = os.path.join(ref_dir, f"{pid}.png")
    srch_path = os.path.join(srch_dir, f"{pid}.png")

    for attempt in range(max_retries):
        # 1. Generate search fine canvas
        zone_res_search = generate_zone_canvas(
            canvas_dim, kind, collapse_threshold_nm=10.0, rng=rng,
            mat_size_nm=2600.0, strip_width_nm=320.0
        )
        search_canvas = zone_res_search["canvas"]
        strip_rects = zone_res_search.get("strip_rects") or []
        mat_rects = zone_res_search.get("mat_rects") or []

        # 2. Warp search image with anti-aliasing
        search_raw = warp_affine_antialiased(
            search_canvas, search_w, search_h, c_canvas, c_search, z, theta, oversample_factor=4
        )

        if present == 1:
            T_s2c = get_search_to_canvas_matrix(c_canvas, c_search, z, theta)
            T_c2s = get_canvas_to_search_matrix(c_canvas, c_search, z, theta)

            chosen = False
            if is_ambiguous and mat_rects and attempt < 3:
                # Sample inside a dense array mat for natural difficulty
                for _ in range(30):
                    m_idx = int(rng.integers(0, len(mat_rects)))
                    mx, my, mw, mh = mat_rects[m_idx]
                    cand_c_canvas = (mx + mw / 2.0 + rng.uniform(-60, 60), my + mh / 2.0 + rng.uniform(-60, 60))
                    p_s = transform_point(cand_c_canvas, T_c2s)
                    if 220.0 <= p_s[0] <= 780.0 and 220.0 <= p_s[1] <= 780.0:
                        c_ref_x, c_ref_y = cand_c_canvas
                        chosen = True
                        break

            if not chosen and strip_rects:
                for _ in range(30):
                    s_idx = int(rng.integers(0, len(strip_rects)))
                    sx, sy, sw, sh = strip_rects[s_idx]
                    cand_c_canvas = (sx + sw / 2.0 + rng.uniform(-80, 80), sy + sh / 2.0 + rng.uniform(-80, 80))
                    p_s = transform_point(cand_c_canvas, T_c2s)
                    if 220.0 <= p_s[0] <= 780.0 and 220.0 <= p_s[1] <= 780.0:
                        c_ref_x, c_ref_y = cand_c_canvas
                        chosen = True
                        break

            if not chosen:
                target_x = float(rng.uniform(250.0, 750.0))
                target_y = float(rng.uniform(250.0, 750.0))
                c_ref_x, c_ref_y = transform_point((target_x, target_y), T_s2c)

            # Crop 1000x1000 reference canvas patch
            x0 = int(round(c_ref_x - (ref_w - 1) / 2.0))
            y0 = int(round(c_ref_y - (ref_h - 1) / 2.0))

            x0 = max(0, min(x0, canvas_dim - ref_w))
            y0 = max(0, min(y0, canvas_dim - ref_h))
            actual_c_ref_x = x0 + (ref_w - 1) / 2.0
            actual_c_ref_y = y0 + (ref_h - 1) / 2.0

            true_x, true_y = transform_point((actual_c_ref_x, actual_c_ref_y), T_c2s)

            ref_canvas_crop = search_canvas[y0:y0 + ref_h, x0:x0 + ref_w].copy()

            # Apply SEM imaging degradations
            ref_degraded = gaussian_psf_blur(ref_canvas_crop, spot_size_nm=5.0, pixel_size_nm=1.0)
            ref_degraded = add_shot_noise(ref_degraded, dose=2000.0, rng=rng)
            ref_degraded = add_detector_noise(ref_degraded, sigma=2.0, rng=rng)

            search_degraded, shift_x = apply_severity_degradations(search_raw, severity, rng)
            true_x += shift_x

            gt_present = 1
            gt_x = round(true_x, 4)
            gt_y = round(true_y, 4)
            gt_theta = round(theta, 4)
            gt_scale = round(z, 4)

        else:
            # Set C: Absent pair
            # Authentic intra-family decoy with non-matching macro-structure
            zone_res_decoy = generate_zone_canvas(
                canvas_dim, kind, collapse_threshold_nm=10.0, rng=rng,
                mat_size_nm=4200.0, strip_width_nm=650.0
            )
            decoy_canvas = zone_res_decoy["canvas"]

            # Distinctive central macro alignment pad absent in search image
            cx_decoy = int(canvas_dim // 2)
            cy_decoy = int(canvas_dim // 2)
            cv2.rectangle(decoy_canvas, (cx_decoy - 300, cy_decoy - 300), (cx_decoy + 300, cy_decoy + 300), 210, -1)
            cv2.circle(decoy_canvas, (cx_decoy, cy_decoy), 180, 50, -1)
            cv2.line(decoy_canvas, (cx_decoy - 480, cy_decoy), (cx_decoy + 480, cy_decoy), 255, 30)
            cv2.line(decoy_canvas, (cx_decoy, cy_decoy - 480), (cx_decoy, cy_decoy + 480), 255, 30)

            dx0 = int(cx_decoy - ref_w // 2)
            dy0 = int(cy_decoy - ref_h // 2)

            ref_canvas_crop = decoy_canvas[dy0:dy0 + ref_h, dx0:dx0 + ref_w].copy()
            ref_degraded = gaussian_psf_blur(ref_canvas_crop, spot_size_nm=5.0, pixel_size_nm=1.0)
            ref_degraded = add_shot_noise(ref_degraded, dose=2000.0, rng=rng)
            ref_degraded = add_detector_noise(ref_degraded, sigma=2.0, rng=rng)

            search_degraded, _ = apply_severity_degradations(search_raw, severity=0, rng=rng)

            gt_present = 0
            gt_x = 0.0
            gt_y = 0.0
            gt_theta = 0.0
            gt_scale = 0.0

        # Set D Optical Microscope Simulation
        if set_type == "D":
            ref_final = simulate_optical_image(ref_degraded, rng)
            search_final = simulate_optical_image(search_degraded, rng)
        else:
            ref_final = ref_degraded
            search_final = search_degraded

        # Write to disk
        cv2.imwrite(ref_path, ref_final)
        cv2.imwrite(srch_path, search_final)

        # Non-negotiable Disk Read-Back Verification Gate (Section 5)
        # Margin floor: 0.02 for high severity Set B, 0.03 for others
        margin_floor = 0.02 if (set_type == "B" and severity >= 3) else 0.03
        verif = verify_disk_pair(srch_path, ref_path, gt_present, gt_x, gt_y, gt_theta, gt_scale, margin_floor)

        if verif["verified"]:
            return {
                "pair_id": pid,
                "set": set_type,
                "preset": preset_name,
                "kind": kind,
                "present": gt_present,
                "x": gt_x,
                "y": gt_y,
                "theta": gt_theta,
                "scale": gt_scale,
                "search_path": f"search/{pid}.png",
                "reference_path": f"reference/{pid}.png",
                "severity_level": severity,
                "v1_peak": verif.get("v1_peak", 1.0),
                "v1_margin": verif.get("v1_margin", 1.0),
                "v1_error": verif.get("v1_error", 0.0),
                "v2_error": verif.get("v2_error", 0.0),
                "attempts": attempt + 1
            }

    raise RuntimeError(f"FATAL: Pair {pid} failed verification gate after {max_retries} retries! Last reason: {verif['reason']}")


def main():
    parser = argparse.ArgumentParser(description="Drift-Sense Phase 2 Dataset Generator")
    parser.add_argument("--output-dir", default="output", help="Directory to save dataset and metadata")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for reproducibility")
    parser.add_argument("--pairs", type=int, default=20, help="Number of pairs to generate (20 or 200)")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"================================================================================")
    print(f"             DRIFT-SENSE PHASE 2 — DATASET GENERATOR (Seed: {args.seed})         ")
    print(f"================================================================================")
    print(f"Target Output Directory: {output_dir}")
    print(f"Generating {args.pairs} pairs across 12 presets...")

    specs = build_pair_specifications(args.pairs, rng)
    manifest_rows = []
    ground_truth_rows = []
    pairs_rows = []

    t_start = time.perf_counter()
    for idx, spec in enumerate(specs):
        t0 = time.perf_counter()
        res = generate_single_pair(spec, output_dir, rng)
        elapsed = time.perf_counter() - t0

        manifest_rows.append(res)
        ground_truth_rows.append({
            "pair_id": res["pair_id"],
            "present": res["present"],
            "x": res["x"],
            "y": res["y"],
            "theta": res["theta"],
            "scale": res["scale"]
        })
        pairs_rows.append({
            "pair_id": res["pair_id"],
            "search_path": res["search_path"],
            "reference_path": res["reference_path"]
        })

        if (idx + 1) % 5 == 0 or (idx + 1) == len(specs):
            print(f"[{idx+1:03d}/{len(specs):03d}] Generated & Verified {res['pair_id']} (Set {res['set']}, {res['preset']}, z={spec['z']}, th={spec['theta']}, time: {elapsed:.2f}s)")

    # 1. Write solver-facing pairs.csv
    pairs_csv_path = os.path.join(output_dir, "pairs.csv")
    with open(pairs_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "search_path", "reference_path"])
        writer.writeheader()
        writer.writerows(pairs_rows)

    # 2. Write ground_truth.csv (withheld)
    gt_csv_path = os.path.join(output_dir, "ground_truth.csv")
    with open(gt_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "present", "x", "y", "theta", "scale"])
        writer.writeheader()
        writer.writerows(ground_truth_rows)

    # 3. Write manifest.csv and manifest_jury.csv (internal audit / jury sheet)
    manifest_csv_path = os.path.join(output_dir, "manifest.csv")
    with open(manifest_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    jury_csv_path = os.path.join(output_dir, "manifest_jury.csv")
    with open(jury_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    total_time = time.perf_counter() - t_start
    print(f"\n✅ All {len(specs)} pairs successfully generated and certified by verification gate!")
    print(f"Total Generation Time: {total_time:.2f}s (Average: {total_time/len(specs):.2f}s/pair)")
    print(f"Emitted Files:")
    print(f"  • Solver CSV:      {pairs_csv_path}")
    print(f"  • Ground Truth:    {gt_csv_path}")
    print(f"  • Audit Manifest:  {manifest_csv_path}")


if __name__ == "__main__":
    main()
