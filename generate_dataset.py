#!/usr/bin/env python3
"""
DriftSense — Standalone Wafer Dataset Generator.

Generates procedurally created dense repeating Manhattan-style semiconductor wafer
patterns with tiled unit cells, regular pitches, scribe lines, and architecture-specific
structures:
1. DRAM: Staggered arrays of capacitor ovals with metal line traces and contact vias.
2. FinFET: Arrays of thin vertical fins crossed by horizontal gates with contact pads.

Simulates SEM degradations: Gaussian noise, multiplicative speckle noise, defocus blur, 
and localized electro-static charging gradients. Logs the true pattern center as ground truth.
"""

import os
import sys
import json
import csv
import math
import argparse
import numpy as np
import cv2
from typing import Any, Tuple, Dict, Optional

# Standard configuration defaults if config.yaml is not present
DEFAULT_CONFIG = {
    "seed": 42,
    "search_size": [512, 512],
    "reference_size": [256, 256],
    "zoom_ratio": 5.0,
    "pattern_densities": [0.1, 0.4],
    "rotation_bounds": [-3.0, 3.0],
    "scale_bounds": [0.97, 1.03],
    "noise_range": [0.01, 0.06],
    "speckle_range": [0.01, 0.05],
    "blur_range": [0.5, 1.5],
    "charging_amplitude_range": [40, 90]
}

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Loads configuration settings from YAML if available, otherwise returns defaults."""
    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Failed to load config.yaml ({e}). Using default parameters.")
    return DEFAULT_CONFIG

def draw_unit_cell(
    style: str,
    pitch_x: int,
    pitch_y: int,
    density: float,
    bg_val: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Renders a unit cell patch of either DRAM or FinFET layout with procedural noise.
    
    Args:
        style: Semiconductor architecture style ("DRAM" or "FinFET").
        pitch_x: Width of the unit cell patch in pixels.
        pitch_y: Height of the unit cell patch in pixels.
        density: feature density control.
        bg_val: Background grayscale intensity (silicon substrate).
        rng: Random generator.
    """
    cell = np.full((pitch_y, pitch_x), bg_val, dtype=np.float32)
    
    if style.lower() == "dram":
        # Staggered capacitor arrays
        cx = pitch_x // 2
        cy = pitch_y // 2
        
        # Determine oval dimensions based on pitch and random scaling
        axes_x_center = max(2, int(pitch_x * rng.uniform(0.12, 0.22)))
        axes_y_center = max(4, int(pitch_y * rng.uniform(0.20, 0.32)))
        
        # Center capacitor
        cap_val = rng.uniform(160.0, 240.0)
        cv2.ellipse(cell, (cx, cy), (axes_x_center, axes_y_center), 0.0, 0.0, 360.0, cap_val, -1)
        
        # Staggered corner capacitors (four corners mapped together)
        axes_x_corner = max(2, int(pitch_x * rng.uniform(0.12, 0.22)))
        axes_y_corner = max(4, int(pitch_y * rng.uniform(0.20, 0.32)))
        cap_val_corner = rng.uniform(160.0, 240.0)
        for corner_x, corner_y in [(0, 0), (pitch_x, 0), (0, pitch_y), (pitch_x, pitch_y)]:
            cv2.ellipse(cell, (corner_x, corner_y), (axes_x_corner, axes_y_corner), 0.0, 0.0, 360.0, cap_val_corner, -1)
            
        # Add thin wordlines/bitlines passing through
        line_val = rng.uniform(70.0, 120.0)
        t_line = max(1, rng.integers(1, 4))
        cv2.line(cell, (0, cy), (pitch_x, cy), line_val, t_line)
        cv2.line(cell, (cx, 0), (cx, pitch_y), line_val, t_line)
        
        # Draw central contact vias
        via_val = rng.uniform(220.0, 255.0)
        via_w = max(1, rng.integers(2, 5))
        cv2.rectangle(cell, (cx - via_w, cy - via_w), (cx + via_w, cy + via_w), via_val, -1)
        
    elif style.lower() == "finfet":
        # Thin vertical fins crossed by horizontal gates
        num_fins = 3
        fin_spacing = pitch_x // (num_fins + 1)
        fin_val = rng.uniform(180.0, 220.0)
        fin_width = max(1, rng.integers(2, 4))
        
        fin_positions = []
        for f in range(1, num_fins + 1):
            f_x = f * fin_spacing + rng.integers(-2, 3)
            cv2.line(cell, (f_x, 0), (f_x, pitch_y), fin_val, fin_width)
            fin_positions.append(f_x)
            
        num_gates = 2
        gate_spacing = pitch_y // (num_gates + 1)
        gate_val = rng.uniform(120.0, 160.0)
        gate_width = max(1, rng.integers(4, 7))
        
        gate_positions = []
        for g in range(1, num_gates + 1):
            g_y = g * gate_spacing + rng.integers(-2, 3)
            cv2.line(cell, (0, g_y), (pitch_x, g_y), gate_val, gate_width)
            gate_positions.append(g_y)
            
        # Draw contact vias at the intersections
        via_val = rng.uniform(230.0, 255.0)
        via_w = max(1, rng.integers(2, 5))
        for fx in fin_positions:
            for gy in gate_positions:
                if rng.random() < 0.8:
                    cv2.rectangle(cell, (fx - via_w, gy - via_w), (fx + via_w, gy + via_w), via_val, -1)
                    
    else:
        raise ValueError(f"Unknown architecture style: {style}")
        
    return cell

def generate_wafer_canvas(
    width: int,
    height: int,
    density: float,
    style: str,
    rng: np.random.Generator
) -> np.ndarray:
    """Procedurally tiles the wafer pattern canvas."""
    bg_val = rng.uniform(30.0, 50.0)
    canvas = np.full((height, width), bg_val, dtype=np.float32)

    # Determine cell pitch based on density parameter
    pitch = int(120 + (1.0 - density) * 100)
    x_pitch = pitch
    y_pitch = pitch

    for y_start in range(0, height, y_pitch):
        for x_start in range(0, width, x_pitch):
            cell_patch = draw_unit_cell(style, x_pitch, y_pitch, density, bg_val, rng)
            
            # Minor spatial jitter & intensity variation per cell tile
            jx = rng.integers(-3, 4)
            jy = rng.integers(-3, 4)
            intensity_scale = rng.uniform(0.92, 1.08)

            ty, tx = y_start + jy, x_start + jx
            y0_src, y1_src = 0, y_pitch
            x0_src, x1_src = 0, x_pitch
            y0_dst, y1_dst = ty, ty + y_pitch
            x0_dst, x1_dst = tx, tx + x_pitch
            
            if y0_dst < 0:
                y0_src -= y0_dst
                y0_dst = 0
            if y1_dst > height:
                y1_src -= (y1_dst - height)
                y1_dst = height
            if x0_dst < 0:
                x0_src -= x0_dst
                x0_dst = 0
            if x1_dst > width:
                x1_src -= (x1_dst - width)
                x1_dst = width
                
            if (y1_src > y0_src) and (x1_src > x0_src):
                patch = cell_patch[y0_src:y1_src, x0_src:x1_src] * intensity_scale
                canvas[y0_dst:y1_dst, x0_dst:x1_dst] = np.clip(patch, 0.0, 255.0)

    # Scribe lines occurring every 4 tiles
    scribe_spacing_x = x_pitch * 4
    scribe_spacing_y = y_pitch * 4
    scribe_val = rng.uniform(190.0, 230.0)
    scribe_thickness = rng.integers(12, 24)

    for x in range(0, width, scribe_spacing_x):
        cv2.line(canvas, (x, 0), (x, height), scribe_val, scribe_thickness)
    for y in range(0, height, scribe_spacing_y):
        cv2.line(canvas, (0, y), (width, y), scribe_val, scribe_thickness)

    return canvas

def extract_transformed_patch(
    canvas: np.ndarray,
    center: Tuple[float, float],
    size: Tuple[int, int],
    angle_deg: float,
    scale: float
) -> np.ndarray:
    """Extracts a patch with rotation and scale around a center point."""
    cx, cy = center
    w, h = size
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)
    M[0, 2] += (w / 2.0) - cx
    M[1, 2] += (h / 2.0) - cy
    patch = cv2.warpAffine(
        canvas, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
    return patch

def apply_charging_effect(
    image: np.ndarray,
    amplitude: float,
    rng: np.random.Generator
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Applies simulated electro-static charging brightness gradient."""
    h, w = image.shape
    cx = rng.uniform(w * 0.1, w * 0.9)
    cy = rng.uniform(h * 0.1, h * 0.9)
    radius = rng.uniform(w * 0.3, w * 0.8)

    y_indices, x_indices = np.indices((h, w), dtype=np.float32)
    dist_sq = (x_indices - cx) ** 2 + (y_indices - cy) ** 2
    charging_profile = amplitude * np.exp(-dist_sq / (2 * (radius ** 2)))
    params = {
        "cx": float(cx),
        "cy": float(cy),
        "radius": float(radius),
        "amplitude": float(amplitude)
    }
    return np.clip(image + charging_profile, 0.0, 255.0), params

def apply_poisson_gaussian_noise(
    image: np.ndarray,
    poisson_scale: float,
    gaussian_std: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Applies signal-dependent Poisson noise mixed with Gaussian thermal noise.
    I_noisy = Poisson(I * alpha) / alpha + Normal(0, sigma^2)
    where alpha (poisson_scale) maps intensity to electron counts.
    """
    img = image.astype(np.float32)
    if poisson_scale > 0.0:
        scaled = np.maximum(img * poisson_scale, 0.0)
        noisy = rng.poisson(scaled).astype(np.float32) / poisson_scale
    else:
        noisy = img
    if gaussian_std > 0.0:
        gauss = rng.normal(0.0, gaussian_std, img.shape).astype(np.float32)
        noisy = noisy + gauss
    return np.clip(noisy, 0.0, 255.0)

def apply_degradations(
    image: np.ndarray,
    noise_std: float,
    speckle_std: float,
    blur_sigma: float,
    charging_amp: float,
    rng: np.random.Generator,
    use_poisson: bool = True
) -> Tuple[np.ndarray, Optional[Dict[str, float]]]:
    """Applies Gaussian/Poisson noise, speckle noise, blur, and charging degradations."""
    img = image.copy()
    charging_params = None

    if blur_sigma > 0.0:
        ksize = int(round(blur_sigma * 3.0) * 2 + 1)
        ksize = max(3, ksize)
        img = cv2.GaussianBlur(img, (ksize, ksize), blur_sigma)

    if abs(charging_amp) > 0.0:
        img, charging_params = apply_charging_effect(img, charging_amp, rng)

    if use_poisson:
        # Map noise_std (typically 0.01 - 0.06) to poisson_scale. 
        # Lower noise_std -> higher poisson_scale (more electrons, less shot noise)
        # We can define poisson_scale = 1.0 / (noise_std ** 2) as a physical relation
        p_scale = 1.0 / max(1e-5, noise_std ** 2) if noise_std > 0.0 else 0.0
        # Thermal Gaussian noise scales with noise_std as well
        g_std = noise_std * 127.5 # up to ~8 gray levels at 0.06
        img = apply_poisson_gaussian_noise(img, p_scale, g_std, rng)
    else:
        if noise_std > 0.0:
            gauss_noise = rng.normal(0.0, noise_std * 255.0, img.shape).astype(np.float32)
            img = img + gauss_noise

    if speckle_std > 0.0:
        speckle_noise = rng.normal(0.0, speckle_std, img.shape).astype(np.float32)
        img = img * (1.0 + speckle_noise)

    img = np.clip(img, 0.0, 255.0)
    return img.astype(np.uint8), charging_params

def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone DriftSense Wafer Dataset Generator.")
    parser.add_argument("--style", type=str, choices=["DRAM", "FinFET"], required=True,
                        help="Wafer architecture design style (DRAM or FinFET).")
    parser.add_argument("--num_pairs", type=int, default=5,
                        help="Number of reference+search image pairs to generate.")
    parser.add_argument("--output_dir", type=str, default="data/generated",
                        help="Output directory path where dataset will be saved.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for procedural generation reproducibility.")
    parser.add_argument("--phase2", action="store_true", default=True,
                        help="Generate Phase 2 compliant dataset (1000x1000 px, [8,12] scale, ±5 deg, absent pairs).")
    parser.add_argument("--absent_prob", type=float, default=0.20,
                        help="Probability of generating an absent pair (found = 0).")
    args = parser.parse_args()

    print(f"Generating {args.num_pairs} wafer pattern pairs for style '{args.style}' (Phase 2: {args.phase2})...")
    config = load_config("config.yaml")

    rng = np.random.default_rng(args.seed)
    
    if args.phase2:
        search_h, search_w = 1000, 1000
        ref_h, ref_w = 256, 256
        zoom_ratio = 10.0
        rot_min, rot_max = -5.0, 5.0
        scale_min, scale_max = 0.8, 1.2 # yields final scale in [8.0, 12.0]
    else:
        search_h, search_w = config.get("search_size", [512, 512])
        ref_h, ref_w = config.get("reference_size", [256, 256])
        zoom_ratio = config.get("zoom_ratio", 5.0)
        rot_min, rot_max = config.get("rotation_bounds", [-3.0, 3.0])
        scale_min, scale_max = config.get("scale_bounds", [0.97, 1.03])

    # Establish parameter bounds
    noise_min, noise_max = config.get("noise_range", [0.01, 0.06])
    speckle_min, speckle_max = config.get("speckle_range", [0.01, 0.05])
    blur_min, blur_max = config.get("blur_range", [0.5, 1.5])
    charge_min, charge_max = config.get("charging_amplitude_range", [40, 90])
    density_min, density_max = config.get("pattern_densities", [0.1, 0.4])

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "ground_truth.csv")
    csv_rows = []

    for i in range(args.num_pairs):
        sample_density = rng.uniform(density_min, density_max)
        sample_noise = rng.uniform(noise_min, noise_max)
        sample_speckle = rng.uniform(speckle_min, speckle_max)
        sample_blur = rng.uniform(blur_min, blur_max)
        sample_charge = rng.uniform(charge_min, charge_max) * rng.choice([-1.0, 1.0])
        
        sample_rot = rng.uniform(rot_min, rot_max)
        sample_scale = rng.uniform(scale_min, scale_max)

        # Determine if this pair is absent (found = 0)
        is_absent = (rng.uniform() < args.absent_prob) if args.phase2 else False

        # Allocate canvas space to extract patches without borders
        search_canvas_w = int(search_w * zoom_ratio)
        search_canvas_h = int(search_h * zoom_ratio)
        canvas_w = search_canvas_w + 1000
        canvas_h = search_canvas_h + 1000

        # Generate unique procedurally tiled canvas for the search image
        canvas_search = generate_wafer_canvas(canvas_w, canvas_h, sample_density, args.style, rng)

        # Randomize search crop coordinate center
        search_cx = rng.uniform(search_canvas_w / 2.0, canvas_w - search_canvas_w / 2.0)
        search_cy = rng.uniform(search_canvas_h / 2.0, canvas_h - search_canvas_h / 2.0)

        # Extract search region
        search_crop_canvas = extract_transformed_patch(
            canvas_search, center=(search_cx, search_cy), size=(search_canvas_w, search_canvas_h), angle_deg=0.0, scale=1.0
        )
        search_img_clean = cv2.resize(search_crop_canvas, (search_w, search_h), interpolation=cv2.INTER_AREA)

        # If absent, generate a COMPLETELY separate canvas with same style but different density to represent another die region
        if is_absent:
            ref_density = rng.uniform(density_min, density_max)
            # Generate a different canvas
            canvas_ref = generate_wafer_canvas(canvas_w, canvas_h, ref_density, args.style, rng)
            # Pick a center on the reference canvas
            ref_cx = rng.uniform(search_canvas_w / 2.0, canvas_w - search_canvas_w / 2.0)
            ref_cy = rng.uniform(search_canvas_h / 2.0, canvas_h - search_canvas_h / 2.0)
            # Ground truths are zero or null for absent pairs
            true_x, true_y = 0.0, 0.0
            scale_factor = 0.0
            rotation_deg = 0.0
            found = 0
        else:
            canvas_ref = canvas_search
            # Randomize reference crop center within search crop bounds (representing stage drift)
            # Drift is typically small (within 5.0 pixels in search image space)
            max_drift_search_px = 5.0
            max_drift_canvas = max_drift_search_px * zoom_ratio
            
            offset_x_canvas = rng.uniform(-max_drift_canvas, max_drift_canvas)
            offset_y_canvas = rng.uniform(-max_drift_canvas, max_drift_canvas)

            ref_cx = search_cx + offset_x_canvas
            ref_cy = search_cy + offset_y_canvas

            # Calculate true center in search image coordinate space
            tl_x_canvas = search_cx - (search_canvas_w / 2.0)
            tl_y_canvas = search_cy - (search_canvas_h / 2.0)
            true_x = (ref_cx - tl_x_canvas) / zoom_ratio
            true_y = (ref_cy - tl_y_canvas) / zoom_ratio
            scale_factor = sample_scale * zoom_ratio
            rotation_deg = sample_rot
            found = 1

        # Extract reference patch
        ref_img_clean = extract_transformed_patch(
            canvas_ref, center=(ref_cx, ref_cy), size=(ref_w, ref_h), angle_deg=sample_rot, scale=sample_scale
        )
        
        ref_img_clean_no_drift = extract_transformed_patch(
            canvas_ref, center=(ref_cx, ref_cy), size=(ref_w, ref_h), angle_deg=0.0, scale=1.0
        )

        # Apply physical electron beam degradations
        search_img, search_charge = apply_degradations(
            search_img_clean, noise_std=sample_noise, speckle_std=sample_speckle,
            blur_sigma=sample_blur, charging_amp=sample_charge, rng=rng
        )
        ref_img, _ = apply_degradations(
            ref_img_clean, noise_std=sample_noise, speckle_std=sample_speckle,
            blur_sigma=sample_blur, charging_amp=sample_charge * 0.5, rng=rng
        )

        # Save files
        sample_dir = os.path.join(args.output_dir, f"sample_{i:03d}")
        os.makedirs(sample_dir, exist_ok=True)

        search_rel_path = os.path.join(sample_dir, "search_image.png")
        ref_rel_path = os.path.join(sample_dir, "reference_image.png")
        clean_rel_path = os.path.join(sample_dir, "reference_clean.png")

        cv2.imwrite(search_rel_path, search_img)
        cv2.imwrite(ref_rel_path, ref_img)
        cv2.imwrite(clean_rel_path, ref_img_clean_no_drift.astype(np.uint8))

        # Output individual JSON metadata
        gt_data = {
            "true_x": float(true_x),
            "true_y": float(true_y),
            "rotation_deg": float(rotation_deg),
            "scale_factor": float(scale_factor),
            "drift_scale": float(sample_scale),
            "zoom_ratio": float(zoom_ratio),
            "noise_level": float(sample_noise),
            "charging_effect": search_charge,
            "style": args.style,
            "found": int(found)
        }
        with open(os.path.join(sample_dir, "ground_truth.json"), "w") as f:
            json.dump(gt_data, f, indent=4)

        # Append to CSV registry
        csv_rows.append({
            "search_image_path": os.path.abspath(search_rel_path),
            "reference_image_path": os.path.abspath(ref_rel_path),
            "GTx": round(true_x, 4),
            "GTy": round(true_y, 4),
            "GT_theta": round(rotation_deg, 4),
            "GT_scale": round(scale_factor, 4),
            "GT_found": int(found),
            "style": args.style
        })

    # Write central CSV registry
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["search_image_path", "reference_image_path", "GTx", "GTy", "GT_theta", "GT_scale", "GT_found", "style"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Successfully generated {args.num_pairs} wafer pairs.")
    print(f"Dataset registry saved to: {os.path.abspath(csv_path)}")

if __name__ == "__main__":
    main()
