import os
import json
import numpy as np
import cv2
from typing import Dict, Any, Tuple, List

from src.data_generation.generate_dataset import generate_wafer_canvas, extract_transformed_patch, apply_degradations

def generate_triplets_for_split(
    split_name: str,
    seed: int,
    num_triplets: int,
    output_dir: str = "data/ml_dataset_v2"
) -> None:
    print(f"Generating {num_triplets} triplets for '{split_name}' split with seed {seed}...")
    rng = np.random.default_rng(seed)
    
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    ref_w, ref_h = 256, 256
    cand_w, cand_h = 256, 256
    zoom_ratio = 5.0
    
    # Degradation parameter ranges (same as config.yaml / ml_config.yaml)
    rot_min, rot_max = -3.0, 3.0
    scale_min, scale_max = 0.97, 1.03
    noise_min, noise_max = 0.01, 0.06
    speckle_min, speckle_max = 0.01, 0.05
    blur_min, blur_max = 0.5, 1.5
    charge_min, charge_max = 40, 90
    density_min, density_max = 0.1, 0.4
    
    metadata = []
    
    # Generate unique canvases per split to avoid leakage
    triplets_per_canvas = 50
    num_canvases = int(np.ceil(num_triplets / triplets_per_canvas))
    
    triplet_idx = 0
    for canvas_i in range(num_canvases):
        canvas_seed = int(rng.integers(0, 100000000))
        canvas_rng = np.random.default_rng(canvas_seed)
        
        density = float(canvas_rng.uniform(density_min, density_max))
        canvas_w, canvas_h = 3000, 3000
        canvas = generate_wafer_canvas(canvas_w, canvas_h, density, canvas_rng)
        
        # Pitch for repeated structures
        pitch = int(120 + (1.0 - density) * 100)
        
        for _ in range(triplets_per_canvas):
            if triplet_idx >= num_triplets:
                break
                
            pair_noise = float(rng.uniform(noise_min, noise_max))
            pair_speckle = float(rng.uniform(speckle_min, speckle_max))
            pair_blur = float(rng.uniform(blur_min, blur_max))
            pair_charge = float(rng.uniform(charge_min, charge_max) * rng.choice([-1.0, 1.0]))
            
            # Select reference center
            margin = max(ref_w, ref_h) * 2
            ref_cx = float(rng.uniform(margin, canvas_w - margin))
            ref_cy = float(rng.uniform(margin, canvas_h - margin))
            
            # True parameters
            sample_rot = float(rng.uniform(rot_min, rot_max))
            sample_scale = float(rng.uniform(scale_min, scale_max))
            
            # 1. Reference Patch
            ref_patch_clean = extract_transformed_patch(
                canvas, (ref_cx, ref_cy), (ref_w, ref_h), sample_rot, sample_scale
            )
            
            # 2. Positive Candidate (with perturbations based on measured classical errors)
            loc_err_x = float(rng.uniform(-1.0, 1.0))
            loc_err_y = float(rng.uniform(-1.0, 1.0))
            rot_err = float(rng.uniform(-1.0, 1.0))
            scale_err = float(rng.uniform(-0.02, 0.02))
            
            pos_cx = ref_cx + loc_err_x * zoom_ratio
            pos_cy = ref_cy + loc_err_y * zoom_ratio
            pos_rot = sample_rot + rot_err
            pos_scale = sample_scale + scale_err
            
            pos_canvas_w = int(cand_w * zoom_ratio * pos_scale)
            pos_canvas_h = int(cand_h * zoom_ratio * pos_scale)
            
            pos_patch_canvas = extract_transformed_patch(
                canvas, (pos_cx, pos_cy), (pos_canvas_w, pos_canvas_h), pos_rot, 1.0
            )
            pos_patch_clean = cv2.resize(
                pos_patch_canvas, (cand_w, cand_h), interpolation=cv2.INTER_AREA
            )
            
            # 3. Hard Negative Candidate
            neg_type = rng.choice(["nearby", "repeated", "wrong_geom", "classical_hard", "random"])
            
            if neg_type == "nearby":
                # Spatially close incorrect center (2.0 to 6.0 search pixels away)
                neg_dx = rng.uniform(2.0, 6.0) * rng.choice([-1, 1])
                neg_dy = rng.uniform(2.0, 6.0) * rng.choice([-1, 1])
                neg_cx = ref_cx + neg_dx * zoom_ratio
                neg_cy = ref_cy + neg_dy * zoom_ratio
                neg_rot = pos_rot
                neg_scale = pos_scale
                
            elif neg_type == "repeated":
                # Shifted by 1 or 2 pitches
                shift_x = int(rng.choice([-2, -1, 1, 2])) * pitch
                shift_y = int(rng.choice([-2, -1, 1, 2])) * pitch
                neg_cx = ref_cx + shift_x
                neg_cy = ref_cy + shift_y
                neg_rot = pos_rot
                neg_scale = pos_scale
                
            elif neg_type == "wrong_geom":
                # Correct center but large rotation/scale offsets
                neg_cx = pos_cx
                neg_cy = pos_cy
                neg_rot = pos_rot + rng.uniform(10.0, 45.0) * rng.choice([-1, 1])
                neg_scale = pos_scale * rng.choice([0.85, 1.15])
                
            elif neg_type == "classical_hard":
                # Classical NMS peak: repeated structure with subpixel location jitter
                shift_x = int(rng.choice([-1, 1])) * pitch
                shift_y = int(rng.choice([-1, 1])) * pitch
                jitter_x = rng.uniform(-1.0, 1.0)
                jitter_y = rng.uniform(-1.0, 1.0)
                neg_cx = ref_cx + shift_x + jitter_x * zoom_ratio
                neg_cy = ref_cy + shift_y + jitter_y * zoom_ratio
                neg_rot = pos_rot
                neg_scale = pos_scale
                
            else:  # "random"
                # Completely unrelated location
                neg_cx = float(rng.uniform(margin, canvas_w - margin))
                neg_cy = float(rng.uniform(margin, canvas_h - margin))
                while np.sqrt((neg_cx - ref_cx)**2 + (neg_cy - ref_cy)**2) < pitch * 3:
                    neg_cx = float(rng.uniform(margin, canvas_w - margin))
                    neg_cy = float(rng.uniform(margin, canvas_h - margin))
                neg_rot = pos_rot
                neg_scale = pos_scale
            
            neg_canvas_w = int(cand_w * zoom_ratio * neg_scale)
            neg_canvas_h = int(cand_h * zoom_ratio * neg_scale)
            
            neg_patch_canvas = extract_transformed_patch(
                canvas, (neg_cx, neg_cy), (neg_canvas_w, neg_canvas_h), neg_rot, 1.0
            )
            neg_patch_clean = cv2.resize(
                neg_patch_canvas, (cand_w, cand_h), interpolation=cv2.INTER_AREA
            )
            
            # Apply degradations to all three patches
            ref_patch, _ = apply_degradations(
                ref_patch_clean, noise_std=pair_noise, speckle_std=pair_speckle,
                blur_sigma=pair_blur, charging_amp=pair_charge * 0.5, rng=rng
            )
            pos_patch, _ = apply_degradations(
                pos_patch_clean, noise_std=pair_noise, speckle_std=pair_speckle,
                blur_sigma=pair_blur, charging_amp=pair_charge, rng=rng
            )
            neg_patch, _ = apply_degradations(
                neg_patch_clean, noise_std=pair_noise, speckle_std=pair_speckle,
                blur_sigma=pair_blur, charging_amp=pair_charge, rng=rng
            )
            
            # Save files
            triplet_id = f"triplet_{triplet_idx:06d}"
            ref_filename = f"{triplet_id}_ref.png"
            pos_filename = f"{triplet_id}_pos.png"
            neg_filename = f"{triplet_id}_neg.png"
            
            cv2.imwrite(os.path.join(split_dir, ref_filename), ref_patch)
            cv2.imwrite(os.path.join(split_dir, pos_filename), pos_patch)
            cv2.imwrite(os.path.join(split_dir, neg_filename), neg_patch)
            
            metadata.append({
                "triplet_id": triplet_id,
                "neg_type": neg_type,
                "source_seed": canvas_seed,
                "ref_loc": [ref_cx, ref_cy],
                "pos_perturb": {
                    "loc_err_x": loc_err_x,
                    "loc_err_y": loc_err_y,
                    "rot_err": rot_err,
                    "scale_err": scale_err
                },
                "ref_path": os.path.join(split_name, ref_filename),
                "pos_path": os.path.join(split_name, pos_filename),
                "neg_path": os.path.join(split_name, neg_filename)
            })
            
            triplet_idx += 1
            if triplet_idx % 1000 == 0:
                print(f"Generated {triplet_idx} / {num_triplets} triplets...")

    metadata_path = os.path.join(output_dir, f"metadata_{split_name}.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Completed {split_name} split. Metadata saved to {metadata_path}\n")

def main() -> None:
    output_dir = "data/ml_dataset_v2"
    generate_triplets_for_split("train", 1000, 5000, output_dir)
    generate_triplets_for_split("val", 2000, 1000, output_dir)
    generate_triplets_for_split("dev", 3000, 1000, output_dir)
    print("V2 Dataset generation completed successfully!")

if __name__ == "__main__":
    main()
