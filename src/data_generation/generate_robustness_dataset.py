"""
Generates the unseen robustness test set of 200 samples using a deterministic seed strategy.
"""
import os
import json
import numpy as np
import cv2
import yaml
from typing import Dict, Any

from src.data_generation.generate_dataset import generate_wafer_canvas, extract_transformed_patch, apply_degradations

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config("config.yaml")
    robustness_master_seed = 50000
    num_samples = 200
    output_dir = "data/robustness_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    search_h, search_w = config["search_size"]
    ref_h, ref_w = config["reference_size"]
    zoom_ratio = config.get("zoom_ratio", 5.0)

    rot_min, rot_max = config["rotation_bounds"]
    scale_min, scale_max = config["scale_bounds"]
    noise_min, noise_max = config["noise_range"]
    speckle_min, speckle_max = config["speckle_range"]
    blur_min, blur_max = config["blur_range"]
    charge_min, charge_max = config["charging_amplitude_range"]
    density_min, density_max = config["pattern_densities"]
    
    print(f"Generating {num_samples} robustness samples in {output_dir}...")
    
    for i in range(num_samples):
        # 1. Deterministic Seeds
        canvas_seed = 5000 + i
        degradation_seed = robustness_master_seed + i
        
        canvas_rng = np.random.default_rng(canvas_seed)
        degradation_rng = np.random.default_rng(degradation_seed)
        
        # 2. Draw parameters using degradation_rng
        sample_density = degradation_rng.uniform(density_min, density_max)
        sample_noise = degradation_rng.uniform(noise_min, noise_max)
        sample_speckle = degradation_rng.uniform(speckle_min, speckle_max)
        sample_blur = degradation_rng.uniform(blur_min, blur_max)
        sample_charge = degradation_rng.uniform(charge_min, charge_max) * degradation_rng.choice([-1.0, 1.0])
        
        sample_rot = degradation_rng.uniform(rot_min, rot_max)
        sample_scale = degradation_rng.uniform(scale_min, scale_max)
        
        search_canvas_w = int(search_w * zoom_ratio)
        search_canvas_h = int(search_h * zoom_ratio)
        
        canvas_w = search_canvas_w + 1000
        canvas_h = search_canvas_h + 1000
        
        # 3. Generate canvas using canvas_rng
        canvas = generate_wafer_canvas(canvas_w, canvas_h, sample_density, canvas_rng)
        
        # 4. Extract search clean patch using degradation_rng (for center selection)
        search_cx = degradation_rng.uniform(search_canvas_w / 2.0, canvas_w - search_canvas_w / 2.0)
        search_cy = degradation_rng.uniform(search_canvas_h / 2.0, canvas_h - search_canvas_h / 2.0)
        
        search_crop_canvas = extract_transformed_patch(
            canvas, center=(search_cx, search_cy), size=(search_canvas_w, search_canvas_h), angle_deg=0.0, scale=1.0
        )
        search_img_clean = cv2.resize(search_crop_canvas, (search_w, search_h), interpolation=cv2.INTER_AREA)
        
        # Offset and reference crop using degradation_rng
        margin_x = ref_w / 2.0
        margin_y = ref_h / 2.0
        max_offset_x = (search_canvas_w / 2.0) - margin_x
        max_offset_y = (search_canvas_h / 2.0) - margin_y
        
        offset_x_canvas = degradation_rng.uniform(-max_offset_x, max_offset_x)
        offset_y_canvas = degradation_rng.uniform(-max_offset_y, max_offset_y)
        
        ref_cx = search_cx + offset_x_canvas
        ref_cy = search_cy + offset_y_canvas
        
        ref_img_clean = extract_transformed_patch(
            canvas, center=(ref_cx, ref_cy), size=(ref_w, ref_h), angle_deg=sample_rot, scale=sample_scale
        )
        
        tl_x_canvas = search_cx - (search_canvas_w / 2.0)
        tl_y_canvas = search_cy - (search_canvas_h / 2.0)
        
        rel_x_canvas = ref_cx - tl_x_canvas
        rel_y_canvas = ref_cy - tl_y_canvas
        
        true_x = rel_x_canvas / zoom_ratio
        true_y = rel_y_canvas / zoom_ratio
        
        # 5. Apply degradations using degradation_rng
        search_img, search_charge = apply_degradations(
            search_img_clean, noise_std=sample_noise, speckle_std=sample_speckle, blur_sigma=sample_blur, charging_amp=sample_charge, rng=degradation_rng
        )
        ref_img, _ = apply_degradations(
            ref_img_clean, noise_std=sample_noise, speckle_std=sample_speckle, blur_sigma=sample_blur, charging_amp=sample_charge * 0.5, rng=degradation_rng
        )
        
        # Save sample files
        sample_dir = os.path.join(output_dir, f"sample_{i:03d}")
        os.makedirs(sample_dir, exist_ok=True)
        
        cv2.imwrite(os.path.join(sample_dir, "search_image.png"), search_img)
        cv2.imwrite(os.path.join(sample_dir, "reference_image.png"), ref_img)
        
        # Calculate boundary proximity for metadata analysis
        dist_left = search_cx
        dist_right = canvas_w - search_cx
        dist_top = search_cy
        dist_bottom = canvas_h - search_cy
        boundary_proximity = min(dist_left, dist_right, dist_top, dist_bottom)
        
        gt_data = {
            "true_x": float(true_x),
            "true_y": float(true_y),
            "rotation_deg": float(sample_rot),
            "scale_factor": float(sample_scale * zoom_ratio),
            "drift_scale": float(sample_scale),
            "zoom_ratio": float(zoom_ratio),
            "noise_level": float(sample_noise),
            "pattern_density": float(sample_density),
            "charging_amplitude": float(sample_charge),
            "boundary_proximity": float(boundary_proximity),
            "canvas_seed": int(canvas_seed),
            "degradation_seed": int(degradation_seed),
            "sample_index": int(i)
        }
        with open(os.path.join(sample_dir, "ground_truth.json"), "w") as f:
            json.dump(gt_data, f, indent=4)
            
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1} / {num_samples}...")

if __name__ == "__main__":
    main()
