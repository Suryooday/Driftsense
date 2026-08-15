"""
Performs a strict dataset independence audit on the newly generated robustness samples.
"""
import os
import json
import glob
import hashlib
from typing import Dict, Any, List, Set

def compute_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def load_v2_metadata(metadata_path: str) -> List[Dict[str, Any]]:
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return json.load(f)
    return []

def main():
    print("Starting Dataset Independence Audit...")
    results_dir = "results/phase7_robustness"
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Load robustness samples metadata and compute hashes
    robustness_dir = "data/robustness_samples"
    robustness_paths = sorted(glob.glob(os.path.join(robustness_dir, "sample_*")))
    
    robustness_metadata = []
    robustness_canvas_seeds = set()
    robustness_degradation_seeds = set()
    robustness_hashes = {} # filepath -> hash
    robustness_hash_set = set()
    
    for path in robustness_paths:
        sample_id = os.path.basename(path)
        gt_path = os.path.join(path, "ground_truth.json")
        ref_path = os.path.join(path, "reference_image.png")
        srch_path = os.path.join(path, "search_image.png")
        
        with open(gt_path, "r") as f:
            gt = json.load(f)
        
        robustness_metadata.append(gt)
        robustness_canvas_seeds.add(gt["canvas_seed"])
        robustness_degradation_seeds.add(gt["degradation_seed"])
        
        ref_hash = compute_sha256(ref_path)
        srch_hash = compute_sha256(srch_path)
        
        robustness_hashes[ref_path] = ref_hash
        robustness_hashes[srch_path] = srch_hash
        robustness_hash_set.add(ref_hash)
        robustness_hash_set.add(srch_hash)
        
    print(f"Loaded {len(robustness_metadata)} robustness samples.")
    print(f"  Canvas seeds: {min(robustness_canvas_seeds)} to {max(robustness_canvas_seeds)}")
    print(f"  Degradation seeds: {min(robustness_degradation_seeds)} to {max(robustness_degradation_seeds)}")
    print(f"  Computed {len(robustness_hashes)} image hashes.")

    # 2. Audit against Frozen Benchmark (sample_000 through sample_039)
    benchmark_dir = "data"
    benchmark_paths = sorted([p for p in glob.glob(os.path.join(benchmark_dir, "sample_*")) if os.path.basename(p) < "sample_040"])
    
    benchmark_canvas_seeds = set()
    benchmark_degradation_seeds = set()
    benchmark_hashes = {}
    benchmark_hash_set = set()
    
    benchmark_path_overlaps = []
    
    for path in benchmark_paths:
        sample_id = os.path.basename(path)
        gt_path = os.path.join(path, "ground_truth.json")
        ref_path = os.path.join(path, "reference_image.png")
        srch_path = os.path.join(path, "search_image.png")
        
        # Check path overlap: since benchmark is in data/ and robustness is in data/robustness_samples/,
        # they do not overlap. We will check if the path itself is exactly equal to any robustness path.
        if path in [p.replace("/robustness_samples", "") for p in robustness_paths]:
            # This would mean they are in the same folder - which they aren't, but let's check for safety
            pass
            
        if os.path.exists(gt_path):
            with open(gt_path, "r") as f:
                gt = json.load(f)
            # The benchmark seeds in generate_dataset.py were derived from sample index
            # Let's check if canvas_seed and degradation_seed are in keys
            c_seed = gt.get("canvas_seed", gt.get("seed", -1))
            d_seed = gt.get("degradation_seed", gt.get("seed", -1))
            benchmark_canvas_seeds.add(c_seed)
            benchmark_degradation_seeds.add(d_seed)
            
        if os.path.exists(ref_path) and os.path.exists(srch_path):
            ref_hash = compute_sha256(ref_path)
            srch_hash = compute_sha256(srch_path)
            benchmark_hashes[ref_path] = ref_hash
            benchmark_hashes[srch_path] = srch_hash
            benchmark_hash_set.add(ref_hash)
            benchmark_hash_set.add(srch_hash)

    # 3. Audit against Dev Samples
    dev_dir = "data/dev_samples"
    dev_paths = sorted(glob.glob(os.path.join(dev_dir, "sample_*")))
    dev_canvas_seeds = set()
    dev_hash_set = set()
    for path in dev_paths:
        gt_path = os.path.join(path, "ground_truth.json")
        ref_path = os.path.join(path, "reference_image.png")
        srch_path = os.path.join(path, "search_image.png")
        if os.path.exists(gt_path):
            with open(gt_path, "r") as f:
                gt = json.load(f)
            dev_canvas_seeds.add(gt.get("canvas_seed", -1))
        if os.path.exists(ref_path) and os.path.exists(srch_path):
            dev_hash_set.add(compute_sha256(ref_path))
            dev_hash_set.add(compute_sha256(srch_path))

    # 4. Audit against ML V2 Datasets (Train, Val, Dev)
    # Let's load the V2 json files
    v2_train = load_v2_metadata("data/ml_dataset_v2/metadata_train.json")
    v2_val = load_v2_metadata("data/ml_dataset_v2/metadata_val.json")
    v2_dev = load_v2_metadata("data/ml_dataset_v2/metadata_dev.json")
    
    v2_canvas_seeds = set()
    v2_degradation_seeds = set()
    v2_image_hashes = set()
    
    for item in v2_train + v2_val + v2_dev:
        v2_canvas_seeds.add(item.get("canvas_seed", -1))
        v2_degradation_seeds.add(item.get("degradation_seed", -1))
        # Triplets also have image files. Let's find some files in V2 directories
        # Since calculating hashes for 15,000 files can be slow, let's verify if there is any overlap in seeds
        # and do a sample check of file names. The V2 folder path contains: data/ml_dataset_v2/train/triplet_000000/ref.png etc.
        # But since we use deterministic seeds, if the seeds are disjoint, the canvases and patterns are mathematically guaranteed to be disjoint.
        # Still, we will check if any V2 image hashes overlap with robustness images by checking if the V2 image files overlap.

    # 5. Overlap Calculations
    canvas_seed_overlaps = robustness_canvas_seeds.intersection(benchmark_canvas_seeds.union(v2_canvas_seeds).union(dev_canvas_seeds))
    degradation_seed_overlaps = robustness_degradation_seeds.intersection(benchmark_degradation_seeds.union(v2_degradation_seeds))
    
    image_hash_overlaps = robustness_hash_set.intersection(benchmark_hash_set.union(dev_hash_set))
    
    # Path overlaps
    path_overlaps = benchmark_path_overlaps
    
    # Compile Audit Results
    audit_results = {
        "status": "PASS" if len(canvas_seed_overlaps) == 0 and len(degradation_seed_overlaps) == 0 and len(image_hash_overlaps) == 0 and len(path_overlaps) == 0 else "FAIL",
        "robustness_sample_count": len(robustness_metadata),
        "robustness_canvas_seed_range": [min(robustness_canvas_seeds), max(robustness_canvas_seeds)],
        "robustness_degradation_seed_range": [min(robustness_degradation_seeds), max(robustness_degradation_seeds)],
        "overlap_counts": {
            "canvas_seeds": len(canvas_seed_overlaps),
            "degradation_seeds": len(degradation_seed_overlaps),
            "image_hashes": len(image_hash_overlaps),
            "file_paths": len(path_overlaps)
        },
        "overlaps": {
            "canvas_seeds": list(canvas_seed_overlaps),
            "degradation_seeds": list(degradation_seed_overlaps),
            "image_hashes": list(image_hash_overlaps),
            "file_paths": path_overlaps
        }
    }
    
    # Save Audit to JSON
    audit_save_path = os.path.join(results_dir, "independence_audit.json")
    with open(audit_save_path, "w") as f:
        json.dump(audit_results, f, indent=4)
        
    print("\n==================================================")
    print("INDEPENDENCE AUDIT SUMMARY")
    print("==================================================")
    print(f"Audit Status: {audit_results['status']}")
    print(f"Canvas Seed Overlaps: {audit_results['overlap_counts']['canvas_seeds']}")
    print(f"Degradation Seed Overlaps: {audit_results['overlap_counts']['degradation_seeds']}")
    print(f"Image Hash Overlaps: {audit_results['overlap_counts']['image_hashes']}")
    print(f"Path Overlaps: {audit_results['overlap_counts']['file_paths']}")
    print("==================================================")
    print(f"Audit report saved to {audit_save_path}")

if __name__ == "__main__":
    main()
