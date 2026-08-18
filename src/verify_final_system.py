"""
Single-command reproducibility check runner.
Verifies config presence, source file paths, SHA-256 hashes, and re-executes the benchmark validation.
"""
import os
import sys
import json
import hashlib
import glob
import subprocess

def compute_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def main():
    print("FINAL SYSTEM VERIFICATION\n")
    
    # 1. Verify final configuration exists
    config_path = "configs/final_system_config.json"
    if os.path.exists(config_path):
        print("Configuration: PASS")
    else:
        print("Configuration: FAIL")
        sys.exit(1)
        
    # 2. Verify required source files exist
    source_files = [
        "src/final_system.py",
        "src/hybrid/candidate_generator.py",
        "src/hybrid/patch_extractor.py",
        "src/matching/classical_matcher.py"
    ]
    source_ok = True
    for sf in source_files:
        if not os.path.exists(sf):
            print(f"  [ERROR] Missing source file: {sf}")
            source_ok = False
    if source_ok:
        print("Source Integrity: PASS")
    else:
        print("Source Integrity: FAIL")
        sys.exit(1)
        
    # 3. Verify benchmark integrity hashes
    hash_manifest_path = "reports/final_freeze/benchmark_hashes.json"
    if not os.path.exists(hash_manifest_path):
        print("Benchmark Hashes: FAIL (Manifest not found)")
        sys.exit(1)
        
    with open(hash_manifest_path, "r") as f:
        manifest = json.load(f)
        
    hashes_ok = True
    for key, expected_hash in manifest.items():
        filepath = os.path.join("data", key)
        if not os.path.exists(filepath):
            print(f"  [ERROR] Missing benchmark file: {filepath}")
            hashes_ok = False
        elif compute_sha256(filepath) != expected_hash:
            print(f"  [ERROR] Hash mismatch for: {filepath}")
            hashes_ok = False
            
    if hashes_ok:
        print("Benchmark Hashes: PASS")
    else:
        print("Benchmark Hashes: FAIL")
        sys.exit(1)
        
    # 4. Run the frozen benchmark validation
    # We can run the existing validation script src/audit/run_benchmark_validation.py via subprocess
    print("Running benchmark validation re-execution...")
    val_script = "src/audit/run_benchmark_validation.py"
    if not os.path.exists(val_script):
        print("Reproducibility: FAIL (Validation script missing)")
        sys.exit(1)
        
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.abspath(".")
    res = subprocess.run(
        [sys.executable, val_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True
    )
    
    if res.returncode != 0:
        print(f"Reproducibility: FAIL (Validation execution crashed: {res.stderr})")
        sys.exit(1)
        
    # Read generated metrics reports/final_freeze/benchmark_metrics.json
    metrics_path = "reports/final_freeze/benchmark_metrics.json"
    if not os.path.exists(metrics_path):
        print("Reproducibility: FAIL (Metrics JSON not written)")
        sys.exit(1)
        
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    # Compare with stored baseline
    expected_success = 0.975
    expected_mean_loc = 0.5425
    expected_median_loc = 0.5280
    expected_mean_rot = 0.0910
    expected_mean_scale = 0.00367
    
    rep_ok = True
    if abs(metrics["success_rate"] - expected_success) > 1e-5:
        print(f"  [ERROR] success_rate mismatch: got {metrics['success_rate']}, expected {expected_success}")
        rep_ok = False
    if abs(metrics["mean_loc"] - expected_mean_loc) > 1e-2:
        print(f"  [ERROR] mean_loc mismatch: got {metrics['mean_loc']:.4f}, expected {expected_mean_loc}")
        rep_ok = False
    if abs(metrics["median_loc"] - expected_median_loc) > 1e-2:
        print(f"  [ERROR] median_loc mismatch: got {metrics['median_loc']:.4f}, expected {expected_median_loc}")
        rep_ok = False
    if abs(metrics["mean_rot"] - expected_mean_rot) > 1e-2:
        print(f"  [ERROR] mean_rot mismatch: got {metrics['mean_rot']:.4f}, expected {expected_mean_rot}")
        rep_ok = False
    if abs(metrics["mean_scale"] - expected_mean_scale) > 1e-3:
        print(f"  [ERROR] mean_scale mismatch: got {metrics['mean_scale']:.5f}, expected {expected_mean_scale}")
        rep_ok = False
        
    if rep_ok:
        print("Reproducibility: PASS")
        print("\nFinal Verdict:")
        print("THE FROZEN SYSTEM REPRODUCES THE VALIDATED RESULTS.")
    else:
        print("Reproducibility: FAIL")
        sys.exit(1)

if __name__ == "__main__":
    main()
