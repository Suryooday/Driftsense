"""
Drift Sense - Alignment Debug Visualizer.

Crops the corresponding region from the search image based on ground truth,
aligns it to the reference image scale/rotation/translation, and displays
them side-by-side to verify ground truth coordinate accuracy.
"""

import os
import json
import glob
import random
import cv2
import matplotlib.pyplot as plt
from typing import List


def verify_alignment(num_samples: int = 5, output_path: str = "data/debug_alignment.png") -> None:
    """
    Crops the matching region from the search image at (true_x, true_y),
    scales it up to the reference image size, and plots them side-by-side.

    Args:
        num_samples: Number of random samples to check.
        output_path: File path to save the debug visualization.
    """
    sample_dirs = sorted(glob.glob("data/sample_*"))
    if not sample_dirs:
        print("No sample directories found in data/. Run generate_dataset.py first.")
        return

    selected_dirs = random.sample(sample_dirs, min(len(sample_dirs), num_samples))
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]

    print(f"Verifying alignment on {len(selected_dirs)} samples...")

    for idx, sample_dir in enumerate(selected_dirs):
        # Load search, reference, and ground truth
        search_img = cv2.imread(os.path.join(sample_dir, "search_image.png"), cv2.IMREAD_GRAYSCALE)
        ref_img = cv2.imread(os.path.join(sample_dir, "reference_image.png"), cv2.IMREAD_GRAYSCALE)
        
        with open(os.path.join(sample_dir, "ground_truth.json"), "r") as f:
            gt = json.load(f)

        true_x = gt["true_x"]
        true_y = gt["true_y"]
        rot = gt["rotation_deg"]
        scale = gt["scale_factor"]
        drift_scale = gt["drift_scale"]
        zoom_ratio = gt["zoom_ratio"]

        # Size of the crop we want to extract from the search image
        # In search pixels, the reference corresponds to a footprint of size (ref_w / scale)
        ref_h, ref_w = ref_img.shape
        w_crop_search = ref_w / scale
        h_crop_search = ref_h / scale

        # We crop around (true_x, true_y) in the search image
        # Since crop sizes must be integers, let's use warpAffine to extract the aligned crop
        # with the exact float coordinates, scale correction, and rotation correction!
        # This undoes the rotation/scale/translation to see if the content matches perfectly.
        M = cv2.getRotationMatrix2D((true_x, true_y), -rot, scale)
        # Shift translation to center the warped region inside a new ref-sized image
        M[0, 2] += (ref_w / 2.0) - true_x
        M[1, 2] += (ref_h / 2.0) - true_y

        aligned_search_crop = cv2.warpAffine(
            search_img, M, (ref_w, ref_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )

        # Plot side-by-side
        ax_ref = axes[idx][0]
        ax_ref.imshow(ref_img, cmap="gray")
        ax_ref.set_title(f"Actual Ref: {os.path.basename(sample_dir)}")
        ax_ref.axis("off")

        ax_crop = axes[idx][1]
        ax_crop.imshow(aligned_search_crop, cmap="gray")
        ax_crop.set_title(f"Aligned Crop from Search")
        ax_crop.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Alignment verification plot saved to: {os.path.abspath(output_path)}")
    plt.close()


if __name__ == "__main__":
    random.seed(42)
    verify_alignment()
