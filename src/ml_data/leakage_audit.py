import os
import json
import hashlib
import numpy as np

def compute_file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def main():
    base_dir = "data/ml_dataset"
    splits = ["train", "val", "dev"]
    
    # Load metadata
    metadata = {}
    for s in splits:
        path = os.path.join(base_dir, f"metadata_{s}.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                metadata[s] = json.load(f)
            print(f"Loaded {s} metadata: {len(metadata[s])} samples.")
        else:
            print(f"Metadata file not found: {path}")
            return

    # 1. Exact file path and seed checks
    print("\n--- Checking Source Seeds (Canvas Seeds) ---")
    seeds = {s: set() for s in splits}
    seed_to_pairs = {s: {} for s in splits}
    
    for s in splits:
        for p in metadata[s]:
            seed = p["source_seed"]
            seeds[s].add(seed)
            if seed not in seed_to_pairs[s]:
                seed_to_pairs[s][seed] = []
            seed_to_pairs[s][seed].append(p["pair_id"])
            
    print(f"Train unique seeds: {len(seeds['train'])}")
    print(f"Val unique seeds: {len(seeds['val'])}")
    print(f"Dev unique seeds: {len(seeds['dev'])}")
    
    # Overlap of seeds
    train_val_seed_overlap = seeds["train"].intersection(seeds["val"])
    train_dev_seed_overlap = seeds["train"].intersection(seeds["dev"])
    val_dev_seed_overlap = seeds["val"].intersection(seeds["dev"])
    
    print(f"Train-Val seed overlap: {len(train_val_seed_overlap)}")
    print(f"Train-Dev seed overlap: {len(train_dev_seed_overlap)}")
    print(f"Val-Dev seed overlap: {len(val_dev_seed_overlap)}")

    # 2. Check for image content hashes (SHA-256)
    print("\n--- Checking Image File Hash Overlaps ---")
    image_hashes = {s: {} for s in splits} # path -> hash
    hash_to_paths = {s: {} for s in splits} # hash -> list of paths
    
    for s in splits:
        print(f"Hashing images in {s} split...")
        for i, p in enumerate(metadata[s]):
            # Paths are relative to data/ml_dataset
            ref_path = os.path.join(base_dir, p["ref_path"])
            cand_path = os.path.join(base_dir, p["cand_path"])
            
            ref_hash = compute_file_hash(ref_path)
            cand_hash = compute_file_hash(cand_path)
            
            image_hashes[s][p["ref_path"]] = ref_hash
            image_hashes[s][p["cand_path"]] = cand_hash
            
            if ref_hash not in hash_to_paths[s]:
                hash_to_paths[s][ref_hash] = []
            hash_to_paths[s][ref_hash].append(p["ref_path"])
            
            if cand_hash not in hash_to_paths[s]:
                hash_to_paths[s][cand_hash] = []
            hash_to_paths[s][cand_hash].append(p["cand_path"])

    # Overlaps in hashes
    all_hashes = {s: set(hash_to_paths[s].keys()) for s in splits}
    
    train_val_hash_overlap = all_hashes["train"].intersection(all_hashes["val"])
    train_dev_hash_overlap = all_hashes["train"].intersection(all_hashes["dev"])
    val_dev_hash_overlap = all_hashes["val"].intersection(all_hashes["dev"])
    
    print(f"Train-Val image hash overlap: {len(train_val_hash_overlap)}")
    print(f"Train-Dev image hash overlap: {len(train_dev_hash_overlap)}")
    print(f"Val-Dev image hash overlap: {len(val_dev_hash_overlap)}")
    
    # Let's inspect detail on hash overlap if any exists
    if len(train_val_hash_overlap) > 0:
        print("Train-Val overlapping hash paths:")
        for h in list(train_val_hash_overlap)[:5]:
            print(f"Hash {h[:8]}: Train={hash_to_paths['train'][h]} | Val={hash_to_paths['val'][h]}")

    # 3. Check for Benchmark Leakage
    print("\n--- Checking Frozen Benchmark Leakage ---")
    benchmark_dir = "data"
    benchmark_hashes = {}
    
    for i in range(40):
        sample_path = os.path.join(benchmark_dir, f"sample_{i:03d}")
        for filename in ["search_image.png", "reference_image.png", "reference_clean.png"]:
            filepath = os.path.join(sample_path, filename)
            if os.path.exists(filepath):
                file_hash = compute_file_hash(filepath)
                benchmark_hashes[file_hash] = filepath
                
    print(f"Computed hashes for {len(benchmark_hashes)} benchmark images.")
    
    # Check if any benchmark image hash is in any split
    leaked_found = False
    for s in splits:
        split_hashes = set(hash_to_paths[s].keys())
        overlap = split_hashes.intersection(benchmark_hashes.keys())
        print(f"Benchmark overlap with {s} split: {len(overlap)}")
        if len(overlap) > 0:
            leaked_found = True
            for h in overlap:
                print(f"  Benchmark file {benchmark_hashes[h]} leaked into split {s}: {hash_to_paths[s][h]}")
                
    if not leaked_found:
        print("--> Confirmed: NO BENCHMARK LEAKAGE DETECTED.")
        
if __name__ == "__main__":
    main()
