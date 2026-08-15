"""
Submission packaging script for SEMICON India Hackathon.
Bundles the codebase, dependencies, dataset generator, localization scripts,
DL training materials, model checkpoints, and technical documentation into a clean ZIP file.
"""
import os
import zipfile

def create_submission_zip():
    zip_filename = "DriftSense_Hackathon_Submission.zip"
    print(f"Creating submission ZIP package: {zip_filename}...")
    
    files_to_include = [
        "README.md",
        "requirements.txt",
        "COLAB_TRAINING.md",
        "predict.py",
        "src/final_system.py",
        "src/drift_recovery.py",
        "src/driftsense.py",
        "src/run_final.py",
        "src/data_generation/generate_dataset.py",
        "src/matching/classical_matcher.py",
        "src/hybrid/candidate_generator.py",
        "src/hybrid/patch_extractor.py",
        "src/ml/train_matcher.py",
        "src/ml/model.py",
        "src/ml_v2/train_matcher_v2.py",
        "src/ml_v2/model_v2.py",
        "src/scoring/evaluate.py",
        "src/scoring/eval_dataset_csv.py",
        "backend/main.py",
        "backend/schemas.py",
        "backend/requirements.txt",
        "backend/services/driftsense_service.py"
    ]
    
    dirs_to_include = [
        "configs",
        "models",
        "reports"
    ]
    
    added_files = set()

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add core files
        for fpath in files_to_include:
            if os.path.exists(fpath) and fpath not in added_files:
                zipf.write(fpath, arcname=fpath)
                added_files.add(fpath)
                print(f"  + Added: {fpath}")
                
        # Add required directories
        for dpath in dirs_to_include:
            for root, dirs, files in os.walk(dpath):
                for file in files:
                    if not file.endswith(".pyc") and file != ".DS_Store":
                        full_path = os.path.join(root, file)
                        if full_path not in added_files:
                            zipf.write(full_path, arcname=full_path)
                            added_files.add(full_path)
                        
    print(f"\nSuccessfully created submission ZIP: {zip_filename} ({os.path.getsize(zip_filename) / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    create_submission_zip()
