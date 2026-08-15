"""
Drift Sense - Synthetic Wafer Dataset Generator.

Generates procedurally created dense repeating Manhattan-style semiconductor wafer
patterns with tiled unit cells, regular pitches, per-tile structural variation (jitter
in line positions, thicknesses, and secondary markers), and scribe line grids.
Simulates SEM (Scanning Electron Microscope) degradations and logs charging parameters.
"""

import os
import json
import yaml
import numpy as np
import cv2
from typing import Any, Tuple, Dict, Optional


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Loads configuration settings from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A dictionary containing configuration parameters.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def draw_unit_cell(
    pitch_x: int,
    pitch_y: int,
    density: float,
    bg_val: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generates a unit cell with unique structural variations per tile, including
    line-position jitter, line-thickness variation, and occasional secondary features.

    Args:
        pitch_x: Width of the unit cell in pixels.
        pitch_y: Height of the unit cell in pixels.
        density: Pattern density parameter.
        bg_val: Grayscale value for the silicon substrate background.
        rng: NumPy random generator instance.

    Returns:
        A 2D float32 numpy array representing the grayscale unit cell patch.
    """
    cell = np.full((pitch_y, pitch_x), bg_val, dtype=np.float32)
    
    # Base block dimensions
    block_w = int(pitch_x * rng.uniform(0.4, 0.65))
    block_h = int(pitch_y * rng.uniform(0.4, 0.65))
    
    # Central block center with slight jitter in position (±1-3px)
    bx_jitter = rng.integers(-3, 4)
    by_jitter = rng.integers(-3, 4)
    bx = (pitch_x - block_w) // 2 + bx_jitter
    by = (pitch_y - block_h) // 2 + by_jitter
    
    block_val = rng.uniform(90.0, 140.0)
    cv2.rectangle(cell, (bx, by), (bx + block_w, by + block_h), block_val, -1)

    # Draw vertical and horizontal metal traces with per-tile jitter in position & thickness
    trace_val = rng.uniform(160.0, 220.0)
    
    # Jitter in trace positions (±1-3px)
    vx1_jit = rng.integers(-3, 4)
    vx2_jit = rng.integers(-3, 4)
    hy1_jit = rng.integers(-3, 4)
    hy2_jit = rng.integers(-3, 4)
    
    v_x1 = bx + int(block_w * 0.25) + vx1_jit
    v_x2 = bx + int(block_w * 0.75) + vx2_jit
    h_y1 = by + int(block_h * 0.25) + hy1_jit
    h_y2 = by + int(block_h * 0.75) + hy2_jit

    # Per-trace jitter in thickness
    t_v1 = max(1, rng.integers(2, 6))
    t_v2 = max(1, rng.integers(2, 6))
    t_h1 = max(1, rng.integers(2, 6))
    t_h2 = max(1, rng.integers(2, 6))

    cv2.line(cell, (v_x1, 0), (v_x1, pitch_y), trace_val, t_v1)
    cv2.line(cell, (v_x2, 0), (v_x2, pitch_y), trace_val, t_v2)
    cv2.line(cell, (0, h_y1), (pitch_x, h_y1), trace_val, t_h1)
    cv2.line(cell, (0, h_y2), (pitch_x, h_y2), trace_val, t_h2)

    # Draw contact vias at intersections
    via_val = rng.uniform(220.0, 255.0)
    via_size = max(2, rng.integers(3, 7))
    for vx in [v_x1, v_x2]:
        for vy in [h_y1, h_y2]:
            cv2.rectangle(
                cell, 
                (vx - via_size // 2, vy - via_size // 2), 
                (vx + via_size // 2, vy + via_size // 2), 
                via_val, 
                -1
            )

    # Occasionally (~12% probability) add a secondary feature (a square contact marker)
    if rng.random() < 0.12:
        marker_size = rng.integers(6, 12)
        mx = bx + rng.choice([0, block_w])
        my = by + rng.choice([0, block_h])
        marker_val = rng.uniform(180.0, 240.0)
        cv2.rectangle(
            cell,
            (mx - marker_size // 2, my - marker_size // 2),
            (mx + marker_size // 2, my + marker_size // 2),
            marker_val,
            -1
        )
        
    return cell


def generate_wafer_canvas(
    width: int,
    height: int,
    density: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Procedurally generates a dense repeating Manhattan-style wafer pattern canvas.
    Tiles unit cells at a regular pitch in X and Y with per-tile structural variation
    and regular horizontal and vertical scribe grid lines.

    Args:
        width: Width of the canvas in pixels.
        height: Height of the canvas in pixels.
        density: Control parameter for grid/feature density (0.0 to 1.0).
        rng: NumPy random generator instance.

    Returns:
        A 2D float32 numpy array representing the grayscale canvas (0.0 to 255.0).
    """
    # Substrate base background intensity
    bg_val = rng.uniform(30.0, 50.0)
    canvas = np.full((height, width), bg_val, dtype=np.float32)

    # Pitch ranges from 120px to 220px on the canvas
    pitch = int(120 + (1.0 - density) * 100)
    x_pitch = pitch
    y_pitch = pitch

    # Tile the unique unit cells across the full canvas
    for y_start in range(0, height, y_pitch):
        for x_start in range(0, width, x_pitch):
            # Generate the unique cell template with structural variation
            cell_patch = draw_unit_cell(x_pitch, y_pitch, density, bg_val, rng)
            
            # Positional jitter of the entire tile block (+/- 2 to 3 pixels)
            jx = rng.integers(-3, 4)
            jy = rng.integers(-3, 4)
            # Minor intensity variation
            intensity_scale = rng.uniform(0.92, 1.08)

            ty = y_start + jy
            tx = x_start + jx

            # Clip template patch to canvas boundaries
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

    # Add periodic horizontal AND vertical boundary/scribe lines (streets) at regular intervals
    # Scribe lines occur every 4 unit cells
    scribe_spacing_x = x_pitch * 4
    scribe_spacing_y = y_pitch * 4
    scribe_val = rng.uniform(190.0, 230.0)
    scribe_thickness = rng.integers(12, 24)

    # Draw vertical scribe lines
    for x in range(0, width, scribe_spacing_x):
        cv2.line(canvas, (x, 0), (x, height), scribe_val, scribe_thickness)

    # Draw horizontal scribe lines
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
    """
    Extracts a patch of specified size centered at `center` with rotation and scale.

    Args:
        canvas: Grayscale canvas float32 array.
        center: (x, y) center coordinate on the canvas.
        size: (width, height) of the output patch.
        angle_deg: Rotation angle in degrees (clockwise).
        scale: Scale factor (relative to 1.0 canvas resolution).

    Returns:
        Extracted and transformed grayscale patch as float32 array.
    """
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
    """
    Applies a localized brightness gradient or vignetting to simulate SEM charging.

    Args:
        image: Input float32 grayscale image.
        amplitude: Peak brightness change (positive or negative).
        rng: NumPy random generator instance.

    Returns:
        A tuple of (Grayscale image with charging effect applied, dict of parameters).
    """
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


def apply_degradations(
    image: np.ndarray,
    noise_std: float,
    speckle_std: float,
    blur_sigma: float,
    charging_amp: float,
    rng: np.random.Generator
) -> Tuple[np.ndarray, Optional[Dict[str, float]]]:
    """
    Simulates SEM degradation effects (Gaussian, speckle, blur, charging)
    and returns the processed image and charging effect parameters if applied.

    Args:
        image: Grayscale float32 input image.
        noise_std: Standard deviation of additive Gaussian noise.
        speckle_std: Standard deviation of multiplicative speckle noise.
        blur_sigma: Gaussian blur standard deviation.
        charging_amp: Peak amplitude of the SEM charging effect.
        rng: NumPy random generator instance.

    Returns:
        A tuple containing:
        - The degraded grayscale image of type uint8.
        - Dict of charging effect parameters (or None if not applied).
    """
    img = image.copy()
    charging_params = None

    # 1. Apply Gaussian Blur
    if blur_sigma > 0.0:
        ksize = int(round(blur_sigma * 3.0) * 2 + 1)
        ksize = max(3, ksize)
        img = cv2.GaussianBlur(img, (ksize, ksize), blur_sigma)

    # 2. Localized SEM Charging Effect
    if abs(charging_amp) > 0.0:
        img, charging_params = apply_charging_effect(img, charging_amp, rng)

    # 3. Additive Gaussian Noise
    if noise_std > 0.0:
        gauss_noise = rng.normal(0.0, noise_std * 255.0, img.shape).astype(np.float32)
        img = img + gauss_noise

    # 4. Multiplicative Speckle Noise
    if speckle_std > 0.0:
        speckle_noise = rng.normal(0.0, speckle_std, img.shape).astype(np.float32)
        img = img * (1.0 + speckle_noise)

    # Clamp and convert to uint8
    img = np.clip(img, 0.0, 255.0)
    return img.astype(np.uint8), charging_params


def main() -> None:
    """Main execution function to generate the synthetic wafer matching dataset."""
    print("Starting Drift Sense dataset generation...")
    config = load_config("config.yaml")

    # Set reproducibility parameters
    seed = config.get("seed", 42)
    rng = np.random.default_rng(seed)

    num_samples = config.get("num_samples", 40)
    search_h, search_w = config["search_size"]
    ref_h, ref_w = config["reference_size"]
    zoom_ratio = config.get("zoom_ratio", 5.0)

    # Output directory
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # Setup ranges from config
    rot_min, rot_max = config["rotation_bounds"]
    scale_min, scale_max = config["scale_bounds"]
    noise_min, noise_max = config["noise_range"]
    speckle_min, speckle_max = config["speckle_range"]
    blur_min, blur_max = config["blur_range"]
    charge_min, charge_max = config["charging_amplitude_range"]
    density_min, density_max = config["pattern_densities"]

    # Statistics accumulators
    noise_levels = []
    pattern_densities = []

    for i in range(num_samples):
        # 1. Randomize parameters for this sample
        sample_density = rng.uniform(density_min, density_max)
        sample_noise = rng.uniform(noise_min, noise_max)
        sample_speckle = rng.uniform(speckle_min, speckle_max)
        sample_blur = rng.uniform(blur_min, blur_max)
        sample_charge = rng.uniform(charge_min, charge_max) * rng.choice([-1.0, 1.0])
        
        sample_rot = rng.uniform(rot_min, rot_max)
        sample_scale = rng.uniform(scale_min, scale_max)

        pattern_densities.append(sample_density)
        noise_levels.append(sample_noise)

        # Calculate canvas size to guarantee crops fit
        search_canvas_w = int(search_w * zoom_ratio)
        search_canvas_h = int(search_h * zoom_ratio)
        
        canvas_w = search_canvas_w + 1000
        canvas_h = search_canvas_h + 1000

        # Generate base clean tiled canvas with unique per-tile variations
        canvas = generate_wafer_canvas(canvas_w, canvas_h, sample_density, rng)

        # 2. Select Search crop center randomly from canvas (ensuring it fits)
        search_cx = rng.uniform(search_canvas_w / 2.0, canvas_w - search_canvas_w / 2.0)
        search_cy = rng.uniform(search_canvas_h / 2.0, canvas_h - search_canvas_h / 2.0)

        # Extract search region from canvas
        search_crop_canvas = extract_transformed_patch(
            canvas,
            center=(search_cx, search_cy),
            size=(search_canvas_w, search_canvas_h),
            angle_deg=0.0,
            scale=1.0
        )
        
        # Resize search image to target resolution
        search_img_clean = cv2.resize(
            search_crop_canvas, (search_w, search_h), interpolation=cv2.INTER_AREA
        )

        # 3. Select Reference crop center randomly within search crop boundaries
        margin_x = ref_w / 2.0
        margin_y = ref_h / 2.0
        
        # Max offset in canvas coordinates
        max_offset_x = (search_canvas_w / 2.0) - margin_x
        max_offset_y = (search_canvas_h / 2.0) - margin_y

        offset_x_canvas = rng.uniform(-max_offset_x, max_offset_x)
        offset_y_canvas = rng.uniform(-max_offset_y, max_offset_y)

        ref_cx = search_cx + offset_x_canvas
        ref_cy = search_cy + offset_y_canvas

        # Extract reference patch from canvas (applies rotation/scale drift)
        ref_img_clean = extract_transformed_patch(
            canvas,
            center=(ref_cx, ref_cy),
            size=(ref_w, ref_h),
            angle_deg=sample_rot,
            scale=sample_scale
        )

        # 4. Calculate Ground Truth pixel location in search image coordinate space
        tl_x_canvas = search_cx - (search_canvas_w / 2.0)
        tl_y_canvas = search_cy - (search_canvas_h / 2.0)

        rel_x_canvas = ref_cx - tl_x_canvas
        rel_y_canvas = ref_cy - tl_y_canvas

        true_x = rel_x_canvas / zoom_ratio
        true_y = rel_y_canvas / zoom_ratio

        # 5. Degrade both images to simulate SEM conditions
        search_img, search_charge = apply_degradations(
            search_img_clean,
            noise_std=sample_noise,
            speckle_std=sample_speckle,
            blur_sigma=sample_blur,
            charging_amp=sample_charge,
            rng=rng
        )

        ref_img, _ = apply_degradations(
            ref_img_clean,
            noise_std=sample_noise,
            speckle_std=sample_speckle,
            blur_sigma=sample_blur,
            charging_amp=sample_charge * 0.5,
            rng=rng
        )

        # Extract a clean, un-degraded, un-drifted reference patch for pixel-by-pixel validation
        ref_img_clean_no_drift = extract_transformed_patch(
            canvas,
            center=(ref_cx, ref_cy),
            size=(ref_w, ref_h),
            angle_deg=0.0,
            scale=1.0
        )

        # 6. Save outputs
        sample_dir = os.path.join(data_dir, f"sample_{i:03d}")
        os.makedirs(sample_dir, exist_ok=True)

        cv2.imwrite(os.path.join(sample_dir, "search_image.png"), search_img)
        cv2.imwrite(os.path.join(sample_dir, "reference_image.png"), ref_img)
        cv2.imwrite(os.path.join(sample_dir, "reference_clean.png"), ref_img_clean_no_drift.astype(np.uint8))

        # Store ground truth metadata
        gt_data = {
            "true_x": float(true_x),
            "true_y": float(true_y),
            "rotation_deg": float(sample_rot),
            "scale_factor": float(sample_scale * zoom_ratio),
            "drift_scale": float(sample_scale),
            "zoom_ratio": float(zoom_ratio),
            "noise_level": float(sample_noise),
            "charging_effect": search_charge,
            "search_physical_dims": [search_canvas_w, search_canvas_h],
            "reference_physical_dims_pre_scale": [ref_w, ref_h]
        }

        with open(os.path.join(sample_dir, "ground_truth.json"), "w") as f:
            json.dump(gt_data, f, indent=4)

    print("--------------------------------------------------")
    print("Dataset generation completed successfully!")
    print(f"Total samples generated: {num_samples}")
    print(f"Average noise level: {np.mean(noise_levels):.4f}")
    print(f"Average pattern density: {np.mean(pattern_densities):.4f}")
    print(f"Data saved to: {os.path.abspath(data_dir)}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()
