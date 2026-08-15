"""
Drift Sense - Dataset Visualization Script.

Loads random samples from the generated synthetic dataset and plots them with
ground-truth locations marked to verify data generation accuracy.
"""

import os
import json
import glob
import random
import cv2
import matplotlib.pyplot as plt
from typing import List


def plot_samples(num_samples_to_plot: int = 3, output_path: str = "data/sanity_check.png") -> None:
    """
    Loads random samples, plots search/reference pairs, and marks ground truth.

    Args:
        num_samples_to_plot: Number of random samples to visualize.
        output_path: File path to save the compiled visualization image.
    """
    sample_dirs = sorted(glob.glob("data/sample_*"))
    if not sample_dirs:
        print("No sample directories found in data/. Please run generate_dataset.py first.")
        return

    selected_dirs = random.sample(sample_dirs, min(len(sample_dirs), num_samples_to_plot))
    
    fig, axes = plt.subplots(num_samples_to_plot, 2, figsize=(10, 4 * num_samples_to_plot))
    if num_samples_to_plot == 1:
        axes = [axes]

    for idx, sample_dir in enumerate(selected_dirs):
        # Load images and ground truth
        search_img = cv2.imread(os.path.join(sample_dir, "search_image.png"), cv2.IMREAD_GRAYSCALE)
        ref_img = cv2.imread(os.path.join(sample_dir, "reference_image.png"), cv2.IMREAD_GRAYSCALE)
        
        with open(os.path.join(sample_dir, "ground_truth.json"), "r") as f:
            gt = json.load(f)

        true_x = gt["true_x"]
        true_y = gt["true_y"]
        rot = gt["rotation_deg"]
        scale = gt["scale_factor"]
        drift_scale = gt["drift_scale"]

        # 1. Plot Search Image with GT marked
        ax_search = axes[idx][0]
        ax_search.imshow(search_img, cmap="gray")
        # Draw red dot at ground truth center
        ax_search.scatter(true_x, true_y, color="red", marker="x", s=100, linewidths=2, label="GT Reference Center")
        
        # Draw approximate footprint box if scale and rotation are known
        # The reference footprint in search is: ref_size / scale_factor
        # Since ref_img has width W_ref, and scale_factor = drift_scale * zoom_ratio
        # footprint_w = ref_img_w / scale_factor
        ref_h, ref_w = ref_img.shape
        footprint_w = ref_w / scale
        footprint_h = ref_h / scale
        
        # Draw approximate footprint bounding box
        rect = plt.Rectangle(
            (true_x - footprint_w / 2, true_y - footprint_h / 2),
            footprint_w,
            footprint_h,
            fill=False,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="Approx Footprint"
        )
        ax_search.add_patch(rect)
        ax_search.set_title(f"Search Image (Sample: {os.path.basename(sample_dir)})")
        ax_search.legend(loc="upper right", fontsize="small")

        # 2. Plot Reference Image
        ax_ref = axes[idx][1]
        ax_ref.imshow(ref_img, cmap="gray")
        ax_ref.set_title(f"Ref (100x): Rot={rot:.2f}°, DriftScale={drift_scale:.3f}")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Sanity check visualization saved to: {os.path.abspath(output_path)}")
    plt.close()


if __name__ == "__main__":
    # Fix the seed of the random module for visualization selection consistency
    random.seed(42)
    plot_samples()
