"""
Runs reproducibility audit checks for the frozen wafer matching system.
"""
import os
import json
import sys
import hashlib
import glob

def compute_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def main():
    print("==================================================")
    print("FINAL REPRODUCIBILITY AUDIT")
    print("==================================================")
    
    # 1. Compute benchmark hashes and write / check benchmark_hashes.json
    print("\nAuditing Frozen Benchmark File Integrity...")
    reports_dir = "reports/final_freeze"
    os.makedirs(reports_dir, exist_ok=True)
    hash_save_path = os.path.join(reports_dir, "benchmark_hashes.json")
    
    computed_hashes = {}
    benchmark_dir = "data"
    benchmark_paths = sorted([p for p in glob.glob(os.path.join(benchmark_dir, "sample_*")) if os.path.basename(p) < "sample_040"])
    
    for path in benchmark_paths:
        sample_id = os.path.basename(path)
        for filename in ["reference_image.png", "search_image.png", "ground_truth.json"]:
            filepath = os.path.join(path, filename)
            if os.path.exists(filepath):
                computed_hashes[f"{sample_id}/{filename}"] = compute_sha256(filepath)
                
    if os.path.exists(hash_save_path):
        print(f"Comparing computed hashes against pre-existing manifest at {hash_save_path}...")
        with open(hash_save_path, "r") as f:
            existing_hashes = json.load(f)
            
        mismatch = False
        for k, v in computed_hashes.items():
            if k not in existing_hashes:
                print(f"  [ERROR] File {k} not in pre-existing manifest!")
                mismatch = True
            elif existing_hashes[k] != v:
                print(f"  [ERROR] SHA-256 Mismatch for {k}!")
                mismatch = True
        if not mismatch:
            print("  [SUCCESS] All benchmark file hashes match the manifest.")
        else:
            print("  [FAIL] Hash verification failed!")
            sys.exit(1)
    else:
        print(f"Writing new SHA-256 manifest to {hash_save_path}...")
        with open(hash_save_path, "w") as f:
            json.dump(computed_hashes, f, indent=4)
        print("  [SUCCESS] SHA-256 manifest written.")

    # 2. Check DL V1/V2 Exclusions
    print("\nAuditing final system imports (DL exclusion check)...")
    if "torch" in sys.modules:
        del sys.modules["torch"]
    if "torchvision" in sys.modules:
        del sys.modules["torchvision"]
        
    try:
        from src.final_system import FinalSystemMatcher
        print("  Successfully imported FinalSystemMatcher.")
    except Exception as e:
        print(f"  [ERROR] Failed to import FinalSystemMatcher: {e}")
        sys.exit(1)
        
    if "torch" in sys.modules or "torchvision" in sys.modules:
        print("  [ERROR] PyTorch or TorchVision was loaded after importing FinalSystemMatcher!")
        sys.exit(1)
    else:
        print("  [SUCCESS] PyTorch/TorchVision are NOT loaded during final inference imports.")

    # 3. Check Ground Truth Access
    print("\nAuditing matcher interface for Ground Truth isolation...")
    import inspect
    matcher_sig = inspect.signature(FinalSystemMatcher.match)
    print(f"  FinalSystemMatcher.match signature: {matcher_sig}")
    
    params = list(matcher_sig.parameters.keys())
    if "gt" in params or "ground_truth" in params:
        print("  [ERROR] Matcher interface accepts ground truth directly!")
        sys.exit(1)
        
    with open("src/final_system.py", "r") as f:
        content = f.read()
    if "ground_truth.json" in content:
        print("  [ERROR] Matcher code contains reference to ground_truth.json!")
        sys.exit(1)
    print("  [SUCCESS] Matcher does not access ground truth during inference.")

    # 4. Check configuration completeness
    print("\nAuditing configuration completeness...")
    config_path = "configs/final_system_config.json"
    with open(config_path, "r") as f:
        cfg = json.load(f)
        
    required_keys = ["zoom_ratio", "reference_size", "search_size", "candidate_generation", "pose_refinement"]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        print(f"  [ERROR] Missing keys in final configuration: {missing}")
        sys.exit(1)
    print("  [SUCCESS] Final configuration contains all required parameters.")
    print("\n==================================================")
    print("ALL REPRODUCIBILITY AUDITS PASSED")
    print("==================================================")

if __name__ == "__main__":
    main()
