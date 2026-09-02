import os
import sys
import csv
import json
import time
import numpy as np
import cv2
from pathlib import Path

# Add project root to python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from generate_dataset import (
    generate_wafer_canvas, 
    extract_transformed_patch, 
    apply_degradations
)

def colorize_optical(gray_img: np.ndarray) -> np.ndarray:
    """Simulates a warm-tinted BGR optical microscope image from grayscale."""
    h, w = gray_img.shape
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    f = gray_img.astype(np.float32) / 255.0
    
    # Base silicon substrate color (dark violet-blue)
    # Target features (bright golden/yellowish metallic oxides)
    bgr[..., 0] = np.clip(80 + f * (40 - 80), 0, 255).astype(np.uint8)     # Blue
    bgr[..., 1] = np.clip(30 + f * (180 - 30), 0, 255).astype(np.uint8)    # Green
    bgr[..., 2] = np.clip(50 + f * (240 - 50), 0, 255).astype(np.uint8)    # Red
    return bgr

def main():
    print("Generating Phase 2 Benchmark Dataset (200 pairs)...")
    
    output_dir = Path("data/phase2_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / "ground_truth.csv"
    csv_rows = []
    
    # Setup random seed for dataset reproducibility
    rng = np.random.default_rng(2026)
    
    # Phase 2 dimensions
    search_h, search_w = 1000, 1000
    ref_h, ref_w = 256, 256
    zoom_ratio = 10.0
    
    search_canvas_w = int(search_w * zoom_ratio)
    search_canvas_h = int(search_h * zoom_ratio)
    canvas_w = search_canvas_w + 1000
    canvas_h = search_canvas_h + 1000
    
    # Bounds
    rot_min, rot_max = -5.0, 5.0
    scale_min, scale_max = 0.8, 1.2 # yields final scale in [8.0, 12.0]
    
    # 200 pairs composition:
    # 0-69: Set A (Nominal)
    # 70-139: Set B (Degraded)
    # 140-179: Set C (Absent)
    # 180-199: Set D (Optical)
    
    for i in range(200):
        # Determine set type
        if i < 70:
            set_type = "A" # Nominal
        elif i < 140:
            set_type = "B" # Degraded
        elif i < 180:
            set_type = "C" # Absent
        else:
            set_type = "D" # Optical
            
        style = "DRAM" if (i % 2 == 0) else "FinFET"
        
        # Draw parameters based on set type
        density = rng.uniform(0.1, 0.4)
        
        if set_type == "A":
            # Nominal: low noise, small blur, low/no charging
            noise = rng.uniform(0.01, 0.03)
            speckle = rng.uniform(0.005, 0.02)
            blur = rng.uniform(0.5, 1.2)
            charge = rng.uniform(10.0, 30.0) * rng.choice([-1.0, 1.0])
        elif set_type == "B":
            # Degraded: higher noise, larger blur, strong charging
            noise = rng.uniform(0.03, 0.08)
            speckle = rng.uniform(0.02, 0.06)
            blur = rng.uniform(1.0, 2.5)
            charge = rng.uniform(40.0, 95.0) * rng.choice([-1.0, 1.0])
        elif set_type == "C":
            # Absent: typical nominal noise but different die regions
            noise = rng.uniform(0.015, 0.04)
            speckle = rng.uniform(0.01, 0.03)
            blur = rng.uniform(0.5, 1.5)
            charge = rng.uniform(20.0, 60.0) * rng.choice([-1.0, 1.0])
        else: # Set D (Optical)
            # Optical: low noise, moderate blur, no charging
            noise = rng.uniform(0.01, 0.02)
            speckle = rng.uniform(0.005, 0.01)
            blur = rng.uniform(0.8, 1.6)
            charge = 0.0 # Optical microscope has no electrostatic charging!
            
        sample_rot = rng.uniform(rot_min, rot_max)
        sample_scale = rng.uniform(scale_min, scale_max)
        
        # Build search canvas
        canvas_search = generate_wafer_canvas(canvas_w, canvas_h, density, style, rng)
        
        # Randomize search crop coordinate center
        search_cx = rng.uniform(search_canvas_w / 2.0, canvas_w - search_canvas_w / 2.0)
        search_cy = rng.uniform(search_canvas_h / 2.0, canvas_h - search_canvas_h / 2.0)
        
        search_img_clean_canvas = extract_transformed_patch(
            canvas_search, center=(search_cx, search_cy), size=(search_canvas_w, search_canvas_h), angle_deg=0.0, scale=1.0
        )
        search_img_clean = cv2.resize(search_img_clean_canvas, (search_w, search_h), interpolation=cv2.INTER_AREA)
        
        is_absent = (set_type == "C")
        
        if is_absent:
            ref_density = rng.uniform(0.1, 0.4)
            canvas_ref = generate_wafer_canvas(canvas_w, canvas_h, ref_density, style, rng)
            ref_cx = rng.uniform(search_canvas_w / 2.0, canvas_w - search_canvas_w / 2.0)
            ref_cy = rng.uniform(search_canvas_h / 2.0, canvas_h - search_canvas_h / 2.0)
            true_x, true_y = 0.0, 0.0
            scale_factor = 0.0
            rotation_deg = 0.0
            found = 0
        else:
            canvas_ref = canvas_search
            # Drift is typically small (within 5.0 pixels in search image space)
            max_drift_search_px = 5.0
            max_drift_canvas = max_drift_search_px * zoom_ratio
            
            offset_x_canvas = rng.uniform(-max_drift_canvas, max_drift_canvas)
            offset_y_canvas = rng.uniform(-max_drift_canvas, max_drift_canvas)
            
            ref_cx = search_cx + offset_x_canvas
            ref_cy = search_cy + offset_y_canvas
            
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
        
        # Apply degradations
        search_img, _ = apply_degradations(
            search_img_clean, noise_std=noise, speckle_std=speckle,
            blur_sigma=blur, charging_amp=charge, rng=rng
        )
        ref_img, _ = apply_degradations(
            ref_img_clean, noise_std=noise, speckle_std=speckle,
            blur_sigma=blur, charging_amp=charge * 0.5, rng=rng
        )
        
        # Set D Optical Microscope Analogue: Colorize from grayscale to RGB
        if set_type == "D":
            search_img_out = colorize_optical(search_img)
            ref_img_out = colorize_optical(ref_img)
        else:
            search_img_out = search_img
            ref_img_out = ref_img
            
        # Save images
        sample_dir = output_dir / f"sample_{i:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        search_rel_path = sample_dir / "search_image.png"
        ref_rel_path = sample_dir / "reference_image.png"
        
        cv2.imwrite(str(search_rel_path), search_img_out)
        cv2.imwrite(str(ref_rel_path), ref_img_out)
        
        # Save individual ground_truth.json
        gt_data = {
            "true_x": float(true_x),
            "true_y": float(true_y),
            "rotation_deg": float(rotation_deg),
            "scale_factor": float(scale_factor),
            "zoom_ratio": float(zoom_ratio),
            "noise_level": float(noise),
            "style": style,
            "set": set_type,
            "found": int(found)
        }
        with open(sample_dir / "ground_truth.json", "w") as f:
            json.dump(gt_data, f, indent=4)
            
        csv_rows.append({
            "pair_id": f"phase2_{i:03d}",
            "search_image_path": str(search_rel_path),
            "reference_image_path": str(ref_rel_path),
            "GTx": round(true_x, 4),
            "GTy": round(true_y, 4),
            "GT_theta": round(rotation_deg, 4),
            "GT_scale": round(scale_factor, 4),
            "GT_found": int(found),
            "set": set_type,
            "style": style
        })
        
        if (i + 1) % 10 == 0:
            print(f"Generated {i+1}/200 pairs...")
            
    # Write ground_truth.csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pair_id", "search_image_path", "reference_image_path", 
            "GTx", "GTy", "GT_theta", "GT_scale", "GT_found", "set", "style"
        ])
        writer.writeheader()
        writer.writerows(csv_rows)
        
    print(f"Phase 2 Benchmark Dataset successfully generated!")
    print(f"Registry saved to: {csv_path.resolve()}")

if __name__ == "__main__":
    main()
