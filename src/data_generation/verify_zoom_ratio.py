"""
Drift Sense - Zoom Ratio Verification.

Prints physical crop sizes and physical ratios for the dataset
to explicitly confirm the zoom relationship between reference and search.
"""

import os
import json
import glob
import numpy as np


def verify_zoom_ratio() -> None:
    """
    Computes and prints physical crop dimensions and ratios across the dataset.
    """
    sample_dirs = sorted(glob.glob("data/sample_*"))
    if not sample_dirs:
        print("No sample directories found in data/.")
        return

    # Aggregate info
    ratios = []
    search_phys_widths = []
    ref_phys_widths = []

    for sample_dir in sample_dirs:
        with open(os.path.join(sample_dir, "ground_truth.json"), "r") as f:
            gt = json.load(f)

        search_phys = gt["search_physical_dims"]
        ref_phys_pre = gt["reference_physical_dims_pre_scale"]
        drift_scale = gt["drift_scale"]

        # The physical size of the reference crop on the canvas is (ref_w / drift_scale)
        ref_phys_w = ref_phys_pre[0] / drift_scale
        ref_phys_h = ref_phys_pre[1] / drift_scale

        search_phys_w = search_phys[0]
        search_phys_h = search_phys[1]

        ratio_w = ref_phys_w / search_phys_w
        ratio_h = ref_phys_h / search_phys_h

        ratios.append((ratio_w + ratio_h) / 2.0)
        search_phys_widths.append(search_phys_w)
        ref_phys_widths.append(ref_phys_w)

    print("=== Zoom Ratio Verification Summary ===")
    print(f"Total Samples analyzed           : {len(sample_dirs)}")
    print(f"Avg Search physical dimensions   : {np.mean(search_phys_widths):.1f} x {np.mean(search_phys_widths):.1f} px on canvas")
    print(f"Avg Reference physical dimensions: {np.mean(ref_phys_widths):.1f} x {np.mean(ref_phys_widths):.1f} px on canvas")
    print(f" resulting average ratio (ref/search): {np.mean(ratios):.4f} (expected ~0.1000 for 10x relationship)")
    print(f"Inverse ratio (search/ref)       : {1.0 / np.mean(ratios):.2f}x (nominal 10.00x)")
    print("========================================")


if __name__ == "__main__":
    verify_zoom_ratio()
