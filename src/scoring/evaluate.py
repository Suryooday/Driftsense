"""
Drift Sense — Universal Scoring Script.

Compares any predictions JSON file (produced by the classical or DL matcher)
against per-sample ground_truth.json files and produces:
  - Per-sample error table (console + CSV)
  - Aggregate stats: mean / median / max per error type
  - Breakdown by noise-level bucket (low / medium / high)
  - Inference speed summary (ms per image)
  - Bar chart (matplotlib): error vs noise bucket (saved to PNG)
  - Worst-N samples flagged for failure analysis

Usage:
    python -m src.scoring.evaluate [options]

Options:
    --predictions  Path to predictions JSON file
                   (default: data/predictions_classical.json)
    --data-dir     Root directory containing sample_* subdirectories
                   (default: data)
    --output-dir   Directory for CSV and PNG output
                   (default: data/evaluation)
    --loc-thresh   Location success threshold in pixels (default: 3.0)
    --rot-thresh   Rotation success threshold in degrees (default: 0.5)
    --scale-thresh Scale-drift success threshold (default: 0.02)
    --worst-n      Number of worst samples to flag (default: 5)
    --tag          Short label used in plot titles / CSV filename
                   (default: derived from predictions filename)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import glob
import statistics
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class SampleResult:
    """All error and metadata for one matched sample."""
    sample_id:    str
    loc_error:    float   # Euclidean distance (px)
    rot_error:    float   # absolute degrees
    scale_error:  float   # absolute drift-scale difference
    loc_ok:       bool
    rot_ok:       bool
    scale_ok:     bool
    all_ok:       bool
    noise_level:  float   # from ground_truth.json
    noise_bucket: str     # "low" / "medium" / "high"
    elapsed_s:    Optional[float]   # per-sample inference time (s)
    confidence:   Optional[float]   # matcher confidence score


# ── Helpers ───────────────────────────────────────────────────────────────────

def _noise_bucket(noise: float) -> str:
    """Assigns a noise level to a named bucket."""
    if noise < 0.025:
        return "low"
    elif noise < 0.045:
        return "medium"
    return "high"


def _rot_error(predicted: float, gt: float) -> float:
    """Shortest-path angular error in degrees (handles 360° wrap-around)."""
    raw = abs(predicted - gt) % 360.0
    return raw if raw <= 180.0 else 360.0 - raw


def _loc_error(px: float, py: float, gtx: float, gty: float) -> float:
    """Euclidean location error in pixels."""
    return math.sqrt((px - gtx) ** 2 + (py - gty) ** 2)


def _agg(values: List[float], label: str, unit: str) -> Dict[str, float]:
    """Compute summary statistics for a list of floats."""
    return {
        "label":  label,
        "unit":   unit,
        "n":      len(values),
        "mean":   statistics.mean(values)   if values else float("nan"),
        "median": statistics.median(values) if values else float("nan"),
        "max":    max(values)               if values else float("nan"),
        "min":    min(values)               if values else float("nan"),
    }


# ── Core evaluation logic ─────────────────────────────────────────────────────

def evaluate(
    predictions_path: str = "data/predictions_classical.json",
    data_dir: str = "data",
    output_dir: str = "data/evaluation",
    loc_thresh: float = 3.0,
    rot_thresh: float = 0.5,
    scale_thresh: float = 0.02,
    worst_n: int = 5,
    tag: str = "",
) -> None:
    """
    Runs the full evaluation pipeline.

    Args:
        predictions_path: Path to predictions JSON file.
        data_dir:         Root directory containing sample_* sub-directories.
        output_dir:       Directory where CSV and PNG outputs are saved.
        loc_thresh:       Maximum allowed location error (px) for success.
        rot_thresh:       Maximum allowed rotation error (°) for success.
        scale_thresh:     Maximum allowed scale-drift error for success.
        worst_n:          Number of worst-performing samples to flag.
        tag:              Short label for filenames and plot titles.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not tag:
        base = os.path.splitext(os.path.basename(predictions_path))[0]
        tag = base.replace("predictions_", "").replace("_", "-")

    # ── Load predictions ───────────────────────────────────────────────────────
    with open(predictions_path) as f:
        raw_preds: Dict[str, dict] = json.load(f)

    # ── Load ground truths ─────────────────────────────────────────────────────
    gt_map: Dict[str, dict] = {}
    for gt_path in glob.glob(os.path.join(data_dir, "sample_*", "ground_truth.json")):
        sid = os.path.basename(os.path.dirname(gt_path))
        with open(gt_path) as f:
            gt_map[sid] = json.load(f)

    common = sorted(set(raw_preds.keys()) & set(gt_map.keys()))
    skipped = [s for s in raw_preds if s not in gt_map]
    if skipped:
        print(f"  [warn] {len(skipped)} prediction(s) have no ground truth: {skipped}")

    # ── Per-sample evaluation ──────────────────────────────────────────────────
    results: List[SampleResult] = []
    failed_samples: List[str] = []

    for sid in common:
        p  = raw_preds[sid]
        gt = gt_map[sid]

        # Skip if matcher returned None for this sample
        if p.get("predicted_x") is None:
            failed_samples.append(sid)
            continue

        px, py = float(p["predicted_x"]),    float(p["predicted_y"])
        pr      = float(p["predicted_rotation"])
        ps      = float(p["predicted_scale"])

        gtx, gty = float(gt["true_x"]),      float(gt["true_y"])
        gtr       = float(gt["rotation_deg"])
        gtz       = float(gt["zoom_ratio"])
        gt_drift  = float(gt["drift_scale"])
        noise     = float(gt["noise_level"])

        loc_e   = _loc_error(px, py, gtx, gty)
        rot_e   = _rot_error(pr, gtr)
        pred_ds = ps / gtz
        scale_e = abs(pred_ds - gt_drift)

        loc_ok   = loc_e   <= loc_thresh
        rot_ok   = rot_e   <= rot_thresh
        scale_ok = scale_e <= scale_thresh
        all_ok   = loc_ok and rot_ok and scale_ok

        results.append(SampleResult(
            sample_id=sid,
            loc_error=loc_e,
            rot_error=rot_e,
            scale_error=scale_e,
            loc_ok=loc_ok,
            rot_ok=rot_ok,
            scale_ok=scale_ok,
            all_ok=all_ok,
            noise_level=noise,
            noise_bucket=_noise_bucket(noise),
            elapsed_s=p.get("elapsed_s"),
            confidence=p.get("confidence_score"),
        ))

    n = len(results)
    n_ok = sum(r.all_ok for r in results)

    loc_vals   = [r.loc_error   for r in results]
    rot_vals   = [r.rot_error   for r in results]
    scale_vals = [r.scale_error for r in results]
    elapsed_ms = [r.elapsed_s * 1000 for r in results if r.elapsed_s is not None]
    total_s    = sum(r.elapsed_s for r in results if r.elapsed_s is not None)

    # ── Console header ─────────────────────────────────────────────────────────
    W = 100
    print("=" * W)
    print(f"  DRIFT SENSE — EVALUATION REPORT  [{tag.upper()}]")
    print("=" * W)
    print(f"  Predictions : {os.path.abspath(predictions_path)}")
    print(f"  Data dir    : {os.path.abspath(data_dir)}")
    print(f"  Samples     : {n} evaluated  |  {len(failed_samples)} matcher failures  |  {len(skipped)} missing GT")
    print(f"  Thresholds  : loc < {loc_thresh} px  |  rot < {rot_thresh}°  |  scale < {scale_thresh}")
    print()

    # ── Per-sample table ───────────────────────────────────────────────────────
    HDR = (f"{'Sample':<12} {'LocErr(px)':>10} {'RotErr(°)':>10} {'ScaleErr':>10} "
           f"{'Noise':>7} {'Bucket':>8} {'Conf':>6} {'Time(ms)':>9}  Status")
    print(HDR)
    print("─" * W)
    for r in results:
        t_ms = f"{r.elapsed_s*1000:.1f}" if r.elapsed_s else "  —"
        conf = f"{r.confidence:.3f}" if r.confidence is not None else "  —"
        status = "✓" if r.all_ok else (
            "✗ rot"   if not r.rot_ok   and r.loc_ok and r.scale_ok else
            "✗ scale" if not r.scale_ok and r.loc_ok and r.rot_ok  else
            "✗ loc"   if not r.loc_ok                              else "✗ multi"
        )
        print(
            f"{r.sample_id:<12} {r.loc_error:>10.4f} {r.rot_error:>10.4f} "
            f"{r.scale_error:>10.5f} {r.noise_level:>7.4f} {r.noise_bucket:>8} "
            f"{conf:>6} {t_ms:>9}  {status}"
        )
    print("─" * W)

    # ── Aggregate stats ────────────────────────────────────────────────────────
    print()
    print("── Aggregate Statistics ─────────────────────────────────────────────────────────────")
    for agg in [
        _agg(loc_vals,   "Location error",  "px"),
        _agg(rot_vals,   "Rotation error",  "°"),
        _agg(scale_vals, "Scale error",     ""),
    ]:
        print(f"  {agg['label']:<20}  min={agg['min']:.4f}{agg['unit']}  "
              f"mean={agg['mean']:.4f}{agg['unit']}  "
              f"median={agg['median']:.4f}{agg['unit']}  "
              f"max={agg['max']:.4f}{agg['unit']}")

    print()
    print("── Success Rate ─────────────────────────────────────────────────────────────────────")
    print(f"  Successful matches : {n_ok} / {n}  ({100 * n_ok / max(n, 1):.1f}%)")
    loc_fails   = [r.sample_id for r in results if not r.loc_ok]
    rot_fails   = [r.sample_id for r in results if not r.rot_ok]
    scale_fails = [r.sample_id for r in results if not r.scale_ok]
    print(f"  Location failures  : {len(loc_fails)}  → {loc_fails or 'none'}")
    print(f"  Rotation failures  : {len(rot_fails)}  → {rot_fails}")
    print(f"  Scale failures     : {len(scale_fails)}  → {scale_fails or 'none'}")

    # ── Speed ──────────────────────────────────────────────────────────────────
    print()
    print("── Inference Speed ──────────────────────────────────────────────────────────────────")
    if elapsed_ms:
        print(f"  Mean per sample : {statistics.mean(elapsed_ms):.1f} ms")
        print(f"  Median          : {statistics.median(elapsed_ms):.1f} ms")
        print(f"  Min / Max       : {min(elapsed_ms):.1f} ms / {max(elapsed_ms):.1f} ms")
        print(f"  Total dataset   : {total_s:.1f} s  ({total_s/60:.2f} min)")
    else:
        print("  No timing data in predictions file.")

    # ── Noise-bucket breakdown ─────────────────────────────────────────────────
    print()
    print("── Breakdown by Noise Level Bucket ─────────────────────────────────────────────────")
    bucket_stats: Dict[str, Dict] = {}
    for bucket in ("low", "medium", "high"):
        br = [r for r in results if r.noise_bucket == bucket]
        if not br:
            continue
        b_loc   = [r.loc_error   for r in br]
        b_rot   = [r.rot_error   for r in br]
        b_scale = [r.scale_error for r in br]
        b_ok    = sum(r.all_ok for r in br)
        bucket_stats[bucket] = {
            "n":          len(br),
            "success":    b_ok,
            "mean_loc":   statistics.mean(b_loc),
            "mean_rot":   statistics.mean(b_rot),
            "mean_scale": statistics.mean(b_scale),
            "max_loc":    max(b_loc),
            "noise_vals": [r.noise_level for r in br],
        }
        print(f"  {bucket.upper():>6}  n={len(br):2d}  "
              f"success={b_ok}/{len(br)}  "
              f"mean_loc={statistics.mean(b_loc):.4f}px  "
              f"mean_rot={statistics.mean(b_rot):.4f}°  "
              f"mean_scale={statistics.mean(b_scale):.5f}  "
              f"max_loc={max(b_loc):.4f}px")

    # ── Worst-N flag ───────────────────────────────────────────────────────────
    print()
    print(f"── Worst {worst_n} Samples (by Location Error) ────────────────────────────────────────")
    worst = sorted(results, key=lambda r: r.loc_error, reverse=True)[:worst_n]
    for i, r in enumerate(worst, 1):
        print(f"  #{i}  {r.sample_id}  locErr={r.loc_error:.4f}px  "
              f"rotErr={r.rot_error:.4f}°  scaleErr={r.scale_error:.5f}  "
              f"noise={r.noise_level:.4f}({r.noise_bucket})")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, f"results_{tag}.csv")
    fieldnames = [
        "sample_id", "loc_error", "rot_error", "scale_error",
        "loc_ok", "rot_ok", "scale_ok", "all_ok",
        "noise_level", "noise_bucket", "elapsed_ms", "confidence_score",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            row["elapsed_ms"] = round(r.elapsed_s * 1000, 2) if r.elapsed_s else None
            row.pop("elapsed_s", None)
            writer.writerow({k: row.get(k) for k in fieldnames})
    print()
    print(f"  CSV saved → {csv_path}")

    # ── Bar chart ──────────────────────────────────────────────────────────────
    _make_bar_chart(bucket_stats, tag, output_dir, loc_thresh, rot_thresh, scale_thresh)

    print("=" * W)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _make_bar_chart(
    bucket_stats: Dict[str, Dict],
    tag: str,
    output_dir: str,
    loc_thresh: float,
    rot_thresh: float,
    scale_thresh: float,
) -> None:
    """
    Saves a grouped bar chart of mean error (location, rotation, scale)
    broken down by noise bucket, with threshold lines overlaid.
    """
    buckets = [b for b in ("low", "medium", "high") if b in bucket_stats]
    if not buckets:
        return

    mean_loc   = [bucket_stats[b]["mean_loc"]   for b in buckets]
    mean_rot   = [bucket_stats[b]["mean_rot"]    for b in buckets]
    mean_scale = [bucket_stats[b]["mean_scale"]  for b in buckets]
    max_loc    = [bucket_stats[b]["max_loc"]     for b in buckets]
    success_pct = [
        100 * bucket_stats[b]["success"] / bucket_stats[b]["n"]
        for b in buckets
    ]

    x = np.arange(len(buckets))
    bar_w = 0.22

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Drift Sense Evaluation — {tag.upper()}\nError vs Noise Level Bucket",
        fontsize=13, fontweight="bold", y=1.02,
    )

    palette = ["#4C9BE8", "#F4A942", "#E05252"]
    bucket_colors = {"low": "#6BCB77", "medium": "#FFD93D", "high": "#FF6B6B"}

    # ── Subplot 1: Location error ─────────────────────────────────
    ax = axes[0]
    bars = ax.bar(x, mean_loc, bar_w * 2.5,
                  color=[bucket_colors[b] for b in buckets],
                  edgecolor="white", linewidth=0.8, alpha=0.9, label="Mean")
    ax.bar(x, max_loc, bar_w * 2.5,
           color=[bucket_colors[b] for b in buckets],
           edgecolor="white", linewidth=0.8, alpha=0.35, label="Max")
    ax.axhline(loc_thresh, color="crimson", linestyle="--", linewidth=1.2,
               label=f"Threshold ({loc_thresh} px)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b.capitalize()}\n(n={bucket_stats[b]['n']})" for b in buckets])
    ax.set_ylabel("Location Error (px)")
    ax.set_title("Location Error")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    for bar, val in zip(bars, mean_loc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # ── Subplot 2: Rotation error ─────────────────────────────────
    ax = axes[1]
    bars = ax.bar(x, mean_rot, bar_w * 2.5,
                  color=[bucket_colors[b] for b in buckets],
                  edgecolor="white", linewidth=0.8, alpha=0.9)
    ax.axhline(rot_thresh, color="crimson", linestyle="--", linewidth=1.2,
               label=f"Threshold ({rot_thresh}°)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b.capitalize()}\n(n={bucket_stats[b]['n']})" for b in buckets])
    ax.set_ylabel("Rotation Error (°)")
    ax.set_title("Rotation Error")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    for bar, val in zip(bars, mean_rot):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.3f}°", ha="center", va="bottom", fontsize=8)

    # ── Subplot 3: Success rate ───────────────────────────────────
    ax = axes[2]
    bars = ax.bar(x, success_pct, bar_w * 2.5,
                  color=[bucket_colors[b] for b in buckets],
                  edgecolor="white", linewidth=0.8, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b.capitalize()}\n(n={bucket_stats[b]['n']})" for b in buckets])
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate by Noise Bucket")
    ax.set_ylim(0, 110)
    ax.axhline(100, color="gray", linestyle=":", linewidth=0.8)
    for bar, val in zip(bars, success_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Legend patches for bucket colours
    legend_patches = [mpatches.Patch(color=bucket_colors[b], label=b.capitalize())
                      for b in buckets]
    fig.legend(handles=legend_patches, loc="lower center", ncol=len(buckets),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    chart_path = os.path.join(output_dir, f"chart_{tag}.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {chart_path}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Drift Sense evaluation script")
    p.add_argument("--predictions",  default="data/predictions_classical.json",
                   help="Path to predictions JSON file")
    p.add_argument("--data-dir",     default="data",
                   help="Root directory containing sample_* subdirs")
    p.add_argument("--output-dir",   default="data/evaluation",
                   help="Output directory for CSV and PNG")
    p.add_argument("--loc-thresh",   type=float, default=3.0)
    p.add_argument("--rot-thresh",   type=float, default=0.5)
    p.add_argument("--scale-thresh", type=float, default=0.02)
    p.add_argument("--worst-n",      type=int,   default=5)
    p.add_argument("--tag",          default="",
                   help="Short label for output filenames and titles")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        predictions_path=args.predictions,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        loc_thresh=args.loc_thresh,
        rot_thresh=args.rot_thresh,
        scale_thresh=args.scale_thresh,
        worst_n=args.worst_n,
        tag=args.tag,
    )
