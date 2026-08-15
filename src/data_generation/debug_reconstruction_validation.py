"""
Drift Sense - Reference Image Reconstruction and Ground Truth Validation.

Loads 10 random samples, reconstructs the reference image from its clean,
untransformed base crop using the exact rotation and scale drift parameters,
and computes quantitative similarity metrics (MAE, RMSE, Max Error, SSIM).
"""

import os
import json
import glob
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from typing import Dict, Any, Tuple


def validate_reconstruction(num_samples: int = 10) -> None:
    """
    Validates reference image reconstruction against ground truth for random samples,
    computes quantitative metrics, and saves debug visualization plots.
    """
    sample_dirs = sorted(glob.glob("data/sample_*"))
    if not sample_dirs:
        print("No sample directories found in data/.")
        return

    random.seed(42)
    selected_dirs = random.sample(sample_dirs, min(len(sample_dirs), num_samples))

    print("=== Reference Reconstruction and GT Validation ===")
    print(f"{'Sample':<12} | {'MAE':<7} | {'RMSE':<7} | {'Max Error':<9} | {'SSIM':<7} | {'Status':<6}")
    print("-" * 65)

    os.makedirs("data/debug_reconstruction", exist_ok=True)

    for sample_dir in selected_dirs:
        sample_name = os.path.basename(sample_dir)
        
        # Load files
        ref_clean = cv2.imread(os.path.join(sample_dir, "reference_clean.png"), cv2.IMREAD_GRAYSCALE)
        ref_actual = cv2.imread(os.path.join(sample_dir, "reference_image.png"), cv2.IMREAD_GRAYSCALE)
        
        with open(os.path.join(sample_dir, "ground_truth.json"), "r") as f:
            gt = json.load(f)

        rot = gt["rotation_deg"]
        drift_scale = gt["drift_scale"]
        h_ref, w_ref = ref_clean.shape

        # Reconstruct reference_image by applying rotation and scale drift to reference_clean
        center_ref = (w_ref / 2.0, h_ref / 2.0)
        M = cv2.getRotationMatrix2D(center_ref, rot, drift_scale)
        ref_reconstructed = cv2.warpAffine(
            ref_clean, M, (w_ref, h_ref), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )

        # Convert to float for comparison
        actual_f = ref_actual.astype(float)
        recon_f = ref_reconstructed.astype(float)

        # Calculate metrics
        abs_diff = np.abs(recon_f - actual_f)
        mae = np.mean(abs_diff)
        rmse = np.sqrt(np.mean((recon_f - actual_f) ** 2))
        max_err = np.max(abs_diff)
        
        # Calculate SSIM (Structural Similarity Index)
        score_ssim, _ = ssim(ref_actual, ref_reconstructed, full=True)

        # Status is check based on structural similarity (SSIM should be high despite noise)
        status = "PASS" if score_ssim > 0.4 else "FAIL"

        print(f"{sample_name:<12} | {mae:<7.2f} | {rmse:<7.2f} | {max_err:<9.1f} | {score_ssim:<7.3f} | {status:<6}")

        # Save debug visualization plot
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        axes[0].imshow(ref_clean, cmap="gray")
        axes[0].set_title("Clean Reference Base")
        axes[0].axis("off")

        axes[1].imshow(ref_actual, cmap="gray")
        axes[1].set_title("Actual reference_image.png")
        axes[1].axis("off")

        axes[2].imshow(ref_reconstructed, cmap="gray")
        axes[2].set_title("Reconstructed Ref (No Noise)")
        axes[2].axis("off")

        # Normalize absolute difference image for display
        axes[3].imshow(abs_diff, cmap="hot")
        axes[3].set_title(f"Abs Diff (Max={max_err:.1f})")
        axes[3].axis("off")

        plt.suptitle(f"Sample: {sample_name} | MAE={mae:.2f} | RMSE={rmse:.2f} | SSIM={score_ssim:.3f}")
        plt.tight_layout()
        plt.savefig(os.path.join("data/debug_reconstruction", f"{sample_name}_reconstruction_check.png"), dpi=150)
        plt.close()

    print("-" * 65)
    print(f"Debug reconstruction plots saved to: {os.path.abspath('data/debug_reconstruction')}\n")


if __name__ == "__main__":
    validate_reconstruction()
