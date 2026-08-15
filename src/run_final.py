"""
User-facing command line interface to run the frozen wafer matching system.
Usage:
    python3 -m src.run_final \
        --reference path/to/reference.png \
        --search path/to/search.png \
        --output results/prediction.json
"""
import os
import argparse
import json
import cv2
from src.final_system import FinalSystemMatcher

def main():
    parser = argparse.ArgumentParser(description="Run the frozen final wafer matching system.")
    parser.add_argument("--reference", required=True, help="Path to reference image (PNG)")
    parser.add_argument("--search", required=True, help="Path to search image (PNG)")
    parser.add_argument("--output", required=True, help="Path to save prediction output (JSON)")
    parser.add_argument("--config", default="configs/final_system_config.json", help="Path to system config (JSON)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.reference):
        print(f"Error: Reference image not found at {args.reference}")
        return
    if not os.path.exists(args.search):
        print(f"Error: Search image not found at {args.search}")
        return
        
    ref_img = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
    srch_img = cv2.imread(args.search, cv2.IMREAD_GRAYSCALE)
    
    # Initialize matcher
    matcher = FinalSystemMatcher(config_path=args.config)
    
    # Run matching
    pred_res = matcher.match(ref_img, srch_img)
    
    # Format output
    output_data = {
        "predicted_center": {
            "x": round(pred_res["predicted_x"], 2) if pred_res["predicted_x"] is not None else None,
            "y": round(pred_res["predicted_y"], 2) if pred_res["predicted_y"] is not None else None
        },
        "rotation_degrees": round(pred_res["predicted_rotation"], 2) if pred_res["predicted_rotation"] is not None else None,
        "scale": round(pred_res["predicted_scale"], 3) if pred_res["predicted_scale"] is not None else None,
        "inference_time_seconds": round(pred_res["elapsed_s"], 4)
    }
    
    # Create parent dirs if necessary
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Prediction successfully saved to {args.output}")

if __name__ == "__main__":
    main()
