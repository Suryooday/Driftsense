"""
Drift Sense - Classical Baseline Batch Runner.

Loads every sample from data/sample_*/, runs ClassicalMatcher, and writes
predictions to data/predictions_classical.json in the same per-sample format
as ground_truth.json so it can be directly fed to the scoring pipeline.
"""

from __future__ import annotations

import glob
import json
import os
import time
from typing import Dict, Any

import cv2
import numpy as np
import yaml

from src.matching.classical_matcher import ClassicalMatcher, MatchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Loads the shared YAML config, returning an empty dict on failure."""
    if os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _progress_bar(current: int, total: int, width: int = 40) -> str:
    filled = int(width * current / max(1, total))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {current}/{total}"


# ---------------------------------------------------------------------------
# Main batch runner
# ---------------------------------------------------------------------------

def run_batch(
    data_dir: str = "data",
    output_path: str = "data/predictions_classical.json",
    config_path: str = "config.yaml",
) -> None:
    """
    Iterates over all generated samples, runs the ClassicalMatcher on each,
    and saves a consolidated JSON predictions file.

    Args:
        data_dir: Root directory containing sample_* subdirectories.
        output_path: Destination path for the JSON predictions file.
        config_path: Path to the shared YAML config file.
    """
    cfg = _load_config(config_path)

    # Build matcher from config
    matcher = ClassicalMatcher(
        zoom_ratio=float(cfg.get("zoom_ratio", 5.0)),
        rot_range_deg=tuple(cfg.get("rotation_bounds", [-3.0, 3.0])),  # type: ignore[arg-type]
        scale_range=tuple(cfg.get("scale_bounds", [0.97, 1.03])),      # type: ignore[arg-type]
        rot_steps_coarse=13,
        scale_steps_coarse=7,
        top_k=5,
        nms_radius=20,
        context_scale=1.8,
    )

    sample_dirs = sorted(glob.glob(os.path.join(data_dir, "sample_*")))
    if not sample_dirs:
        print(f"No sample directories found under {data_dir}/")
        return

    n = len(sample_dirs)
    predictions: Dict[str, Any] = {}
    n_success = 0
    total_time_s = 0.0

    print(f"\nDrift Sense — Classical Matcher Baseline")
    print(f"{'─' * 76}")
    print(f"  Samples found : {n}")
    print(f"  Zoom ratio    : {cfg.get('zoom_ratio', 5.0)}")
    print(f"  Rot range     : {cfg.get('rotation_bounds', [-3.0, 3.0])}")
    print(f"  Scale range   : {cfg.get('scale_bounds', [0.97, 1.03])}")
    print(f"{'─' * 76}\n")
    print(f"{'Sample':<12} {'PredX':>8} {'PredY':>8} {'Rot':>7} {'Scale':>7} {'Conf':>6} {'Time(s)':>8}")
    print("─" * 64)

    for idx, sample_dir in enumerate(sample_dirs):
        sample_name = os.path.basename(sample_dir)

        ref_path = os.path.join(sample_dir, "reference_image.png")
        srch_path = os.path.join(sample_dir, "search_image.png")

        if not os.path.exists(ref_path) or not os.path.exists(srch_path):
            print(f"{sample_name:<12} SKIPPED — missing images")
            continue

        reference = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search = cv2.imread(srch_path, cv2.IMREAD_GRAYSCALE)

        if reference is None or search is None:
            print(f"{sample_name:<12} SKIPPED — could not read images")
            continue

        t0 = time.perf_counter()
        result: MatchResult | None = matcher.match(reference, search)
        elapsed = time.perf_counter() - t0
        total_time_s += elapsed

        if result is not None:
            n_success += 1
            pred_dict = {
                "predicted_x": result.predicted_x,
                "predicted_y": result.predicted_y,
                "predicted_rotation": result.predicted_rotation,
                "predicted_scale": result.predicted_scale,
                "confidence_score": result.confidence_score,
                "elapsed_s": round(elapsed, 4),
            }
            print(
                f"{sample_name:<12} "
                f"{result.predicted_x:>8.2f} "
                f"{result.predicted_y:>8.2f} "
                f"{result.predicted_rotation:>7.3f} "
                f"{result.predicted_scale:>7.3f} "
                f"{result.confidence_score:>6.3f} "
                f"{elapsed:>8.3f}s"
            )
        else:
            pred_dict = {
                "predicted_x": None,
                "predicted_y": None,
                "predicted_rotation": None,
                "predicted_scale": None,
                "confidence_score": None,
                "elapsed_s": round(elapsed, 4),
            }
            print(f"{sample_name:<12} FAILED                                {elapsed:>8.3f}s")

        predictions[sample_name] = pred_dict

        # Inline progress bar to stderr
        if (idx + 1) % 5 == 0 or idx == n - 1:
            print(f"  {_progress_bar(idx + 1, n)}", flush=True)

    # ---------------------------------------------------------------
    # Write predictions JSON
    # ---------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=4)

    avg_time = total_time_s / max(1, n)
    print()
    print(f"{'═' * 76}")
    print(f"  Results saved : {os.path.abspath(output_path)}")
    print(f"  Successful    : {n_success} / {n} ({100 * n_success / max(1, n):.1f} %)")
    print(f"  Avg time/img  : {avg_time:.3f} s")
    print(f"  Total elapsed : {total_time_s:.1f} s")
    print(f"{'═' * 76}\n")


if __name__ == "__main__":
    run_batch()
