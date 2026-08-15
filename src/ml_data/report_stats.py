"""
Drift Sense — ML Dataset Statistics Reporter.
"""

import os
import json
import numpy as np
from typing import Dict, Any, List

def analyze_metadata(metadata_path: str) -> Dict[str, Any]:
    with open(metadata_path, "r") as f:
        data = json.load(f)
        
    num_pairs = len(data)
    labels = [p["label"] for p in data]
    positives = sum(labels)
    negatives = num_pairs - positives
    
    neg_types = {}
    for p in data:
        t = p.get("neg_type", "none")
        if t != "none":
            neg_types[t] = neg_types.get(t, 0) + 1
            
    noises = [p["noise_level"] for p in data]
    rotations = [p["rotation_diff"] for p in data]
    scales = [p["scale_diff"] for p in data]
    densities = [p["pattern_density"] for p in data]
    seeds = [p["source_seed"] for p in data]
    
    return {
        "num_pairs": num_pairs,
        "positives": positives,
        "negatives": negatives,
        "neg_types": neg_types,
        "noise_mean": np.mean(noises),
        "noise_std": np.std(noises),
        "rot_mean": np.mean(rotations),
        "rot_std": np.std(rotations),
        "scale_mean": np.mean(scales),
        "scale_std": np.std(scales),
        "density_mean": np.mean(densities),
        "unique_seeds": len(set(seeds)),
        "raw_data": data
    }

def main() -> None:
    dataset_dir = "data/ml_dataset"
    splits = ["train", "val", "dev"]
    
    print("=" * 70)
    print("  DRIFT SENSE — ML DATASET STATISTICS")
    print("=" * 70)
    
    split_stats = {}
    all_seeds = set()
    
    for split in splits:
        metadata_path = os.path.join(dataset_dir, f"metadata_{split}.json")
        if not os.path.exists(metadata_path):
            print(f"Metadata file for split '{split}' not found at {metadata_path}.")
            continue
            
        stats = analyze_metadata(metadata_path)
        split_stats[split] = stats
        
        for p in stats["raw_data"]:
            all_seeds.add(p["source_seed"])
            
        print(f"Split: {split.upper()}")
        print(f"  Total Pairs:        {stats['num_pairs']}")
        print(f"  Positive / Negative: {stats['positives']} / {stats['negatives']} ({stats['positives']/stats['num_pairs']*100:.1f}% positive)")
        print(f"  Hard Negative Types:")
        for t, count in stats["neg_types"].items():
            print(f"    - {t:<12}:    {count}")
        print(f"  Noise level mean:   {stats['noise_mean']:.4f} ± {stats['noise_std']:.4f}")
        print(f"  Rotation diff mean: {stats['rot_mean']:.4f}° ± {stats['rot_std']:.4f}°")
        print(f"  Scale diff mean:    {stats['scale_mean']:.4f} ± {stats['scale_std']:.4f}")
        print(f"  Pattern density:    {stats['density_mean']:.4f}")
        print(f"  Unique seeds:       {stats['unique_seeds']}")
        print("-" * 50)
        
    # Check data leakage with the benchmark dataset
    # Benchmark dataset generated with seed=42
    # Verify that seed 42 is not in any of our splits
    benchmark_seed = 42
    if benchmark_seed in all_seeds:
        print("⚠ ALERT: BENCHMARK SEED (42) WAS DETECTED IN THE ML TRAINING/VAL/DEV SEEDS!")
    else:
        print("✓ SUCCESS: Benchmark seed (42) is isolated and NOT present in the ML dataset.")

if __name__ == "__main__":
    main()
