#!/usr/bin/env python3
"""
DriftSense — Standalone Wafer Pattern Localization Inference Runner.

This script accepts a reference image and a search image, localizes the reference
pattern inside the search image using DriftSense's sub-pixel registration engine,
and prints the localized center coordinates (x, y) to stdout.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
import numpy as np
import cv2

# Set project root on Python path to ensure clean imports of src
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.final_system import FinalSystemMatcher

def main():
    # We allow both positional and keyword arguments for maximum usability
    parser = argparse.ArgumentParser(
        description="Localize a reference wafer pattern inside a search field of view."
    )
    
    # We support positional arguments: localize.py <reference> <search>
    # If the user doesn't pass positional, they can pass keyword: --reference and --search
    parser.add_argument("pos_reference", nargs="?", help="Path to reference image (256x256)")
    parser.add_argument("pos_search", nargs="?", help="Path to search image (512x512)")
    
    parser.add_argument("-r", "--reference", help="Path to reference image (256x256)")
    parser.add_argument("-s", "--search", help="Path to search image (512x512)")
    parser.add_argument("-o", "--output", help="Optional path to save JSON results file")
    parser.add_argument("-c", "--config", default=str(PROJECT_ROOT / "configs" / "final_system_config.json"),
                        help="Path to final system config JSON file")

    args = parser.parse_args()

    # Resolve paths
    ref_path = args.reference or args.pos_reference
    srch_path = args.search or args.pos_search

    if not ref_path or not srch_path:
        print("Error: Missing input image paths.", file=sys.stderr)
        print("Usage:", file=sys.stderr)
        print("  Positional: python3 localize.py path/to/reference.png path/to/search.png", file=sys.stderr)
        print("  Keyword:    python3 localize.py --reference path/to/reference.png --search path/to/search.png", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(ref_path):
        print(f"Error: Reference image not found at '{ref_path}'", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(srch_path):
        print(f"Error: Search image not found at '{srch_path}'", file=sys.stderr)
        sys.exit(1)

    # Load images as grayscale
    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    srch_img = cv2.imread(srch_path, cv2.IMREAD_GRAYSCALE)

    if ref_img is None:
        print(f"Error: Could not load reference image from '{ref_path}' (invalid format?)", file=sys.stderr)
        sys.exit(1)
    if srch_img is None:
        print(f"Error: Could not load search image from '{srch_path}' (invalid format?)", file=sys.stderr)
        sys.exit(1)

    # Initialize matcher
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at '{args.config}'", file=sys.stderr)
        sys.exit(1)
        
    try:
        matcher = FinalSystemMatcher(config_path=args.config)
    except Exception as e:
        print(f"Error: Failed to initialize FinalSystemMatcher: {e}", file=sys.stderr)
        sys.exit(1)

    # Run inference
    t0 = time.perf_counter()
    pred = matcher.match(ref_img, srch_img)
    t_elapsed = time.perf_counter() - t0

    px = pred["predicted_x"]
    py = pred["predicted_y"]

    if px is None or py is None:
        print("Error: Failed to localize pattern in the search image.", file=sys.stderr)
        sys.exit(1)

    # Output to stdout in multiple formats so it fits any parsing scripts
    print(f"(x, y) = ({px:.4f}, {py:.4f})")
    
    # Also write a JSON to stdout for clean parsing
    output_data = {
        "predicted_center": {
            "x": round(px, 4),
            "y": round(py, 4)
        },
        "rotation_degrees": round(pred["predicted_rotation"], 4),
        "scale": round(pred["predicted_scale"], 4),
        "confidence_score": round(pred["confidence_score"], 4),
        "inference_time_seconds": round(t_elapsed, 4)
    }

    # Print JSON representation for structured parsers
    print(json.dumps(output_data, indent=2))

    # Save to file if output is specified
    if args.output:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=4)
            print(f"Results successfully saved to '{args.output}'")
        except Exception as e:
            print(f"Warning: Failed to save output JSON to '{args.output}': {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
