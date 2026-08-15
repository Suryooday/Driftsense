"""
Drift Sense - Charging Effect Debug Visualizer.

Draws a green circle representing the localized SEM charging effect area on the
search image, side-by-side with the original search image, to verify its presence
and visual intensity.
"""

import os
import json
import cv2
import matplotlib.pyplot as plt


def plot_charging_debug(sample_idx: int = 7, output_path: str = "data/debug_charging_effect.png") -> None:
    """
    Plots the search image side-by-side with an annotated version showing the charging effect region.
    """
    sample_dir = f"data/sample_{sample_idx:03d}"
    if not os.path.exists(sample_dir):
        print(f"Sample directory {sample_dir} not found.")
        return

    # Load search image and ground truth
    search_img = cv2.imread(os.path.join(sample_dir, "search_image.png"), cv2.IMREAD_GRAYSCALE)
    with open(os.path.join(sample_dir, "ground_truth.json"), "r") as f:
        gt = json.load(f)

    charging = gt.get("charging_effect")
    if not charging:
        print("No charging effect metadata found in ground truth.")
        return

    cx = charging["cx"]
    cy = charging["cy"]
    radius = charging["radius"]
    amp = charging["amplitude"]

    # Create annotated search image (color image for drawing)
    annotated = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)
    
    # Draw green circle indicating charging area
    cv2.circle(annotated, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)
    # Draw green center cross
    cv2.drawMarker(annotated, (int(cx), int(cy)), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
    
    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(search_img, cmap="gray")
    axes[0].set_title(f"Original Search (Sample: {sample_idx:03d})")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Green Circle: Charging Area (Amp={amp:.1f})")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Charging effect debug visualization saved to: {os.path.abspath(output_path)}")
    plt.close()


if __name__ == "__main__":
    plot_charging_debug()
