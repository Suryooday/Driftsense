"""
Drift Sense — ML Dataset Pair Visualizer.
Plots examples of positives, easy negatives, and hard negatives in a nice layout.
"""

import os
import json
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main() -> None:
    dataset_dir = "data/ml_dataset"
    metadata_path = os.path.join(dataset_dir, "metadata_train.json")
    if not os.path.exists(metadata_path):
        print(f"Train metadata not found at {metadata_path}. Please generate data first.")
        return
        
    with open(metadata_path, "r") as f:
        data = json.load(f)
        
    # Find one example of each type
    examples = {}
    
    # 1. Positive pair
    for p in data:
        if p["label"] == 1:
            examples["positive"] = p
            break
            
    # 2. Easy negative (random unrelated location)
    for p in data:
        if p["label"] == 0 and p["neg_type"] == "random":
            examples["easy_negative"] = p
            break
            
    # 3. Hard negative (nearby spatial drift)
    for p in data:
        if p["label"] == 0 and p["neg_type"] == "nearby":
            examples["hard_negative_nearby"] = p
            break
            
    # 4. Very hard repetitive negative (repeated pitch shift)
    for p in data:
        if p["label"] == 0 and p["neg_type"] == "repeated":
            examples["hard_negative_repetitive"] = p
            break
            
    fig, axes = plt.subplots(4, 2, figsize=(10, 20))
    fig.suptitle("Drift Sense — ML Pair Dataset Visualization", fontsize=16, fontweight="bold")
    
    row_idx = 0
    for name, p in examples.items():
        ref_path = os.path.join(dataset_dir, p["ref_path"])
        cand_path = os.path.join(dataset_dir, p["cand_path"])
        
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        cand_img = cv2.imread(cand_path, cv2.IMREAD_GRAYSCALE)
        
        ax_ref = axes[row_idx, 0]
        ax_cand = axes[row_idx, 1]
        
        ax_ref.imshow(ref_img, cmap="gray")
        ax_ref.set_title(f"{name.upper()} — Reference\n(rot diff={p['rotation_diff']:.2f}°)")
        ax_ref.axis("off")
        
        ax_cand.imshow(cand_img, cmap="gray")
        ax_cand.set_title(f"{name.upper()} — Candidate\n(neg_type={p['neg_type']})")
        ax_cand.axis("off")
        
        row_idx += 1
        
    plt.tight_layout()
    output_png = os.path.join(dataset_dir, "visual_samples.png")
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()
    
    # Also copy to workspace artifact path
    artifact_png = "/Users/suryodaypratapsingh/.gemini/antigravity-ide/brain/57b4bca5-b35d-4153-b946-3670601763e8/visual_samples.png"
    cv2.imwrite(artifact_png, cv2.imread(output_png))
    
    print(f"Pair visualization saved to {output_png} and {artifact_png}")

if __name__ == "__main__":
    main()
