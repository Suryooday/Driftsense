"""
Drift Sense — ML Pair Generation Script.
Procedurally generates positive and hard negative pairs for supervised ML matching.
Strictly separates train, validation, and test datasets.
"""

import os
import json
import yaml
import numpy as np
import cv2
from typing import Dict, Any, Tuple, List

from src.data_generation.generate_dataset import generate_wafer_canvas, extract_transformed_patch, apply_degradations

def load_ml_config(config_path: str = "ml_config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def generate_pairs_for_split(
    split_name: str,
    seed: int,
    num_pairs: int,
    config: Dict[str, Any]
) -> None:
    print(f"Generating {num_pairs} pairs for '{split_name}' split with seed {seed}...")
    rng = np.random.default_rng(seed)
    
    output_dir = config.get("output_dir", "data/ml_dataset")
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    ref_h, ref_w = config["reference_size"]
    cand_h, cand_w = config["candidate_size"]
    zoom_ratio = config.get("zoom_ratio", 5.0)
    
    # Standard boundaries
    rot_min, rot_max = config["rotation_bounds"]
    scale_min, scale_max = config["scale_bounds"]
    noise_min, noise_max = config["noise_range"]
    speckle_min, speckle_max = config["speckle_range"]
    blur_min, blur_max = config["blur_range"]
    charge_min, charge_max = config["charging_amplitude_range"]
    density_min, density_max = config["pattern_densities"]
    
    metadata = []
    
    # We will generate pairs by creating multiple unique canvases
    # To keep canvas generation overhead reasonable while avoiding leakage,
    # we generate 1 base canvas for every 100 pairs, using a unique canvas seed.
    pairs_per_canvas = 100
    num_canvases = int(np.ceil(num_pairs / pairs_per_canvas))
    
    pair_idx = 0
    for canvas_i in range(num_canvases):
        canvas_seed = int(rng.integers(0, 100000000))
        canvas_rng = np.random.default_rng(canvas_seed)
        
        # Random canvas features
        density = float(canvas_rng.uniform(density_min, density_max))
        # Canvas dimensions: large enough to extract multiple independent search and reference regions
        # search region in canvas coords is search_size * zoom_ratio = 512 * 5.0 = 2560
        # Reference patch in canvas coords (pre-scale 1.0) is ref_size = 256x256
        # Let's make the canvas size 4000x4000
        canvas_w, canvas_h = 4000, 4000
        canvas = generate_wafer_canvas(canvas_w, canvas_h, density, canvas_rng)
        
        # Calculate pitch for hard negative structures
        pitch = int(120 + (1.0 - density) * 100)
        
        for _ in range(pairs_per_canvas):
            if pair_idx >= num_pairs:
                break
                
            # Randomize degradation parameters for the pair
            pair_noise = float(rng.uniform(noise_min, noise_max))
            pair_speckle = float(rng.uniform(speckle_min, speckle_max))
            pair_blur = float(rng.uniform(blur_min, blur_max))
            pair_charge = float(rng.uniform(charge_min, charge_max) * rng.choice([-1.0, 1.0]))
            
            # Select reference center (ensuring patch fits within canvas boundaries)
            margin = max(ref_w, ref_h) * 2
            ref_cx = float(rng.uniform(margin, canvas_w - margin))
            ref_cy = float(rng.uniform(margin, canvas_h - margin))
            
            # Determine label: 1 = positive, 0 = negative (50% positive / 50% negative ratio)
            label = int(rng.choice([0, 1]))
            
            # Rotation & Scale bounds
            sample_rot = float(rng.uniform(rot_min, rot_max))
            sample_scale = float(rng.uniform(scale_min, scale_max))
            
            neg_type = "none"
            
            if label == 1:
                # Positive pair: candidate is centered at the same location (in canvas coords)
                # But reference is high-zoom (represents physically smaller area), candidate is low-zoom.
                # Since the reference represents 100x zoom and candidate represents search (10x zoom),
                # reference center corresponds to the center of candidate crop.
                # Let's extract the reference patch (with zoom/drift applied)
                ref_patch_clean = extract_transformed_patch(
                    canvas, (ref_cx, ref_cy), (ref_w, ref_h), sample_rot, sample_scale
                )
                
                # Candidate patch represents the same center, but at search zoom (10x).
                # The search crop is wider and has NO drift (since drift is relative to search).
                # To match search zoom, the candidate patch of size cand_w x cand_h
                # actually spans cand_w * zoom_ratio x cand_h * zoom_ratio on the clean canvas,
                # which is then resized to cand_w x cand_h.
                cand_canvas_w = int(cand_w * zoom_ratio)
                cand_canvas_h = int(cand_h * zoom_ratio)
                
                cand_patch_canvas = extract_transformed_patch(
                    canvas, (ref_cx, ref_cy), (cand_canvas_w, cand_canvas_h), 0.0, 1.0
                )
                cand_patch_clean = cv2.resize(
                    cand_patch_canvas, (cand_w, cand_h), interpolation=cv2.INTER_AREA
                )
                
                delta_x = 0.0
                delta_y = 0.0
                rot_diff = sample_rot
                scale_diff = sample_scale
                
            else:
                # Negative pair: We generate hard negatives
                # Types of hard negatives:
                # A: nearby location (spatial drift / jitter: 10 to 40px away in canvas coordinates)
                # B: repeated structures (shift by exactly 1 or 2 units of pitch)
                # C: incorrect rotation/scale (large rotation/scale offsets, but same center)
                # D: random/unrelated location
                neg_choice = rng.choice(["nearby", "repeated", "wrong_geom", "random"])
                neg_type = neg_choice
                
                if neg_choice == "nearby":
                    # Nearby incorrect location
                    dx = rng.uniform(15.0, 50.0) * rng.choice([-1, 1])
                    dy = rng.uniform(15.0, 50.0) * rng.choice([-1, 1])
                    cand_cx = ref_cx + dx
                    cand_cy = ref_cy + dy
                    
                    ref_patch_clean = extract_transformed_patch(
                        canvas, (ref_cx, ref_cy), (ref_w, ref_h), sample_rot, sample_scale
                    )
                    
                    cand_canvas_w = int(cand_w * zoom_ratio)
                    cand_canvas_h = int(cand_h * zoom_ratio)
                    cand_patch_canvas = extract_transformed_patch(
                        canvas, (cand_cx, cand_cy), (cand_canvas_w, cand_canvas_h), 0.0, 1.0
                    )
                    cand_patch_clean = cv2.resize(
                        cand_patch_canvas, (cand_w, cand_h), interpolation=cv2.INTER_AREA
                    )
                    
                    delta_x = dx / zoom_ratio
                    delta_y = dy / zoom_ratio
                    rot_diff = sample_rot
                    scale_diff = sample_scale
                    
                elif neg_choice == "repeated":
                    # Shift by exactly 1 or 2 repeating grid pitches (simulates highly repetitive match confusion)
                    shift_x = int(rng.choice([-2, -1, 1, 2])) * pitch
                    shift_y = int(rng.choice([-2, -1, 1, 2])) * pitch
                    
                    cand_cx = ref_cx + shift_x
                    cand_cy = ref_cy + shift_y
                    
                    ref_patch_clean = extract_transformed_patch(
                        canvas, (ref_cx, ref_cy), (ref_w, ref_h), sample_rot, sample_scale
                    )
                    
                    cand_canvas_w = int(cand_w * zoom_ratio)
                    cand_canvas_h = int(cand_h * zoom_ratio)
                    cand_patch_canvas = extract_transformed_patch(
                        canvas, (cand_cx, cand_cy), (cand_canvas_w, cand_canvas_h), 0.0, 1.0
                    )
                    cand_patch_clean = cv2.resize(
                        cand_patch_canvas, (cand_w, cand_h), interpolation=cv2.INTER_AREA
                    )
                    
                    delta_x = float(shift_x) / zoom_ratio
                    delta_y = float(shift_y) / zoom_ratio
                    rot_diff = sample_rot
                    scale_diff = sample_scale
                    
                elif neg_choice == "wrong_geom":
                    # Same location, but rotation or scale is significantly wrong (outside matching bounds)
                    # Use a rotation difference between 10 to 45 deg or scale diff of 0.85/1.15
                    wrong_rot = sample_rot + rng.uniform(10.0, 45.0) * rng.choice([-1, 1])
                    wrong_scale = sample_scale * rng.choice([0.85, 1.15])
                    
                    ref_patch_clean = extract_transformed_patch(
                        canvas, (ref_cx, ref_cy), (ref_w, ref_h), wrong_rot, wrong_scale
                    )
                    
                    cand_canvas_w = int(cand_w * zoom_ratio)
                    cand_canvas_h = int(cand_h * zoom_ratio)
                    cand_patch_canvas = extract_transformed_patch(
                        canvas, (ref_cx, ref_cy), (cand_canvas_w, cand_canvas_h), 0.0, 1.0
                    )
                    cand_patch_clean = cv2.resize(
                        cand_patch_canvas, (cand_w, cand_h), interpolation=cv2.INTER_AREA
                    )
                    
                    delta_x = 0.0
                    delta_y = 0.0
                    rot_diff = wrong_rot
                    scale_diff = wrong_scale
                    
                else:
                    # Random completely unrelated location
                    cand_cx = float(rng.uniform(margin, canvas_w - margin))
                    cand_cy = float(rng.uniform(margin, canvas_h - margin))
                    # Avoid overlap
                    while np.sqrt((cand_cx - ref_cx)**2 + (cand_cy - ref_cy)**2) < pitch * 3:
                        cand_cx = float(rng.uniform(margin, canvas_w - margin))
                        cand_cy = float(rng.uniform(margin, canvas_h - margin))
                        
                    ref_patch_clean = extract_transformed_patch(
                        canvas, (ref_cx, ref_cy), (ref_w, ref_h), sample_rot, sample_scale
                    )
                    
                    cand_canvas_w = int(cand_w * zoom_ratio)
                    cand_canvas_h = int(cand_h * zoom_ratio)
                    cand_patch_canvas = extract_transformed_patch(
                        canvas, (cand_cx, cand_cy), (cand_canvas_w, cand_canvas_h), 0.0, 1.0
                    )
                    cand_patch_clean = cv2.resize(
                        cand_patch_canvas, (cand_w, cand_h), interpolation=cv2.INTER_AREA
                    )
                    
                    delta_x = (cand_cx - ref_cx) / zoom_ratio
                    delta_y = (cand_cy - ref_cy) / zoom_ratio
                    rot_diff = sample_rot
                    scale_diff = sample_scale

            # Apply realistic SEM degradations to both patches
            ref_patch, _ = apply_degradations(
                ref_patch_clean,
                noise_std=pair_noise,
                speckle_std=pair_speckle,
                blur_sigma=pair_blur,
                charging_amp=pair_charge * 0.5,
                rng=rng
            )
            
            cand_patch, _ = apply_degradations(
                cand_patch_clean,
                noise_std=pair_noise,
                speckle_std=pair_speckle,
                blur_sigma=pair_blur,
                charging_amp=pair_charge,
                rng=rng
            )
            
            # Save files
            pair_id = f"pair_{pair_idx:06d}"
            ref_filename = f"{pair_id}_ref.png"
            cand_filename = f"{pair_id}_cand.png"
            
            cv2.imwrite(os.path.join(split_dir, ref_filename), ref_patch)
            cv2.imwrite(os.path.join(split_dir, cand_filename), cand_patch)
            
            metadata.append({
                "pair_id": pair_id,
                "label": label,
                "neg_type": neg_type,
                "source_seed": canvas_seed,
                "ref_loc": [ref_cx, ref_cy],
                "delta_x": delta_x,
                "delta_y": delta_y,
                "rotation_diff": rot_diff,
                "scale_diff": scale_diff,
                "noise_level": pair_noise,
                "pattern_density": density,
                "ref_path": os.path.join(split_name, ref_filename),
                "cand_path": os.path.join(split_name, cand_filename)
            })
            
            pair_idx += 1
            if pair_idx % 1000 == 0:
                print(f"Generated {pair_idx} / {num_pairs} pairs...")

    # Save metadata JSON file
    metadata_path = os.path.join(output_dir, f"metadata_{split_name}.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Completed {split_name} split. Metadata saved to {metadata_path}\n")


def main() -> None:
    config = load_ml_config()
    
    # Strictly separated split generators using different seeds
    generate_pairs_for_split("train", config["train_seed"], config["num_train"], config)
    generate_pairs_for_split("val", config["val_seed"], config["num_val"], config)
    generate_pairs_for_split("dev", config["dev_seed"], config["num_dev"], config)
    
    print("All datasets generated successfully!")

if __name__ == "__main__":
    main()
