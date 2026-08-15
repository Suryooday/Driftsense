"""
Generates drift recovery visual diagrams showing expected vs detected targets and correction arrows.
"""
import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_drift_plot(
    sample_id: str,
    search_img: np.ndarray,
    detected: dict,
    expected: dict,
    dx: float,
    dy: float,
    magnitude: float,
    status: str,
    output_path: str
):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(search_img, cmap='gray')
    
    # Plot markers
    ax.plot(expected["x"], expected["y"], 'co', label='Expected Site', markersize=8, markeredgewidth=2)
    ax.plot(detected["x"], detected["y"], 'o', color='#ff6600', label='Detected Site', markersize=8)
    
    # Draw arrow from detected to expected
    # Only draw arrow if magnitude is large enough to see
    if magnitude > 0.5:
        # Use annotate with arrowprops for high-quality arrowhead
        ax.annotate(
            "", 
            xy=(expected["x"], expected["y"]), 
            xytext=(detected["x"], detected["y"]),
            arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle="->", lw=2.5, shrinkA=0, shrinkB=2)
        )
        
    ax.set_title(f"Wafer Inspection Navigation Drift Recovery - {status}", fontsize=14, weight='bold')
    ax.legend(loc='upper right', fontsize=10)
    
    # Info text overlay
    info_text = (
        f"Sample ID: {sample_id}\n"
        f"Expected: X = {expected['x']:.2f}, Y = {expected['y']:.2f}\n"
        f"Detected: X = {detected['x']:.2f}, Y = {detected['y']:.2f}\n"
        f"Drift:\n"
        f"  ΔX = {dx:+.2f} px\n"
        f"  ΔY = {dy:+.2f} px\n"
        f"  Magnitude = {magnitude:.3f} px\n"
        f"Status: {status}\n\n"
        f"Recommended Stage Correction:\n"
        f"  MOVE X: {dx:+.2f} px\n"
        f"  MOVE Y: {dy:+.2f} px"
    )
    
    # Position text box in lower-left corner
    ax.text(
        0.03, 0.03, info_text, 
        transform=ax.transAxes, 
        fontsize=10.5, 
        family='monospace',
        verticalalignment='bottom', 
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
    )
    
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved drift visualization to {output_path}")

def main():
    print("Generating synthetic drift visualizations...")
    figures_dir = "reports/drift_recovery/figures"
    os.makedirs(figures_dir, exist_ok=True)
    
    # Load search image from sample_000
    search_img_path = "data/sample_000/search_image.png"
    if not os.path.exists(search_img_path):
        print(f"Error: Search image not found at {search_img_path}")
        return
    srch = cv2.imread(search_img_path, cv2.IMREAD_GRAYSCALE)
    
    # Detected target from sample_000 prediction (X=446.79, Y=166.34)
    detected = {"x": 446.79, "y": 166.34}
    
    # Scenario 1: ALIGNED
    expected_aligned = {"x": 447.10, "y": 166.50}
    dx_1 = expected_aligned["x"] - detected["x"]
    dy_1 = expected_aligned["y"] - detected["y"]
    mag_1 = np.sqrt(dx_1**2 + dy_1**2)
    generate_drift_plot(
        "sample_000", srch, detected, expected_aligned,
        dx_1, dy_1, mag_1, "ALIGNED",
        os.path.join(figures_dir, "aligned_example.png")
    )
    
    # Scenario 2: MINOR DRIFT
    expected_minor = {"x": 450.00, "y": 168.00}
    dx_2 = expected_minor["x"] - detected["x"]
    dy_2 = expected_minor["y"] - detected["y"]
    mag_2 = np.sqrt(dx_2**2 + dy_2**2)
    generate_drift_plot(
        "sample_000", srch, detected, expected_minor,
        dx_2, dy_2, mag_2, "MINOR_DRIFT",
        os.path.join(figures_dir, "minor_drift_example.png")
    )
    
    # Scenario 3: SIGNIFICANT DRIFT
    expected_sig = {"x": 465.00, "y": 175.00}
    dx_3 = expected_sig["x"] - detected["x"]
    dy_3 = expected_sig["y"] - detected["y"]
    mag_3 = np.sqrt(dx_3**2 + dy_3**2)
    generate_drift_plot(
        "sample_000", srch, detected, expected_sig,
        dx_3, dy_3, mag_3, "SIGNIFICANT_DRIFT",
        os.path.join(figures_dir, "significant_drift_example.png")
    )
    
    print("Synthetic drift visualization generation finished.")

if __name__ == "__main__":
    main()
