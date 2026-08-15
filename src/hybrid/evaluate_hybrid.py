import os
import json
import glob
import time
import argparse
import numpy as np
import cv2
import yaml
from typing import Dict, List, Any, Tuple

from src.hybrid.hybrid_matcher import HybridMatcher
from src.data_generation.generate_dataset import generate_wafer_canvas, extract_transformed_patch, apply_degradations

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def generate_dev_samples(num_samples: int = 20, dev_dir: str = "data/dev_samples") -> None:
    """
    Generates a set of development samples separate from the benchmark to tune the hybrid pipeline.
    """
    if os.path.exists(dev_dir) and len(glob.glob(os.path.join(dev_dir, "sample_*"))) == num_samples:
        print(f"Development samples already exist in {dev_dir}. Skipping generation.")
        return

    print(f"Generating {num_samples} development samples in {dev_dir} using seed 9999...")
    os.makedirs(dev_dir, exist_ok=True)
    
    config = load_config("config.yaml")
    rng = np.random.default_rng(9999) # Separate seed

    search_h, search_w = config["search_size"]
    ref_h, ref_w = config["reference_size"]
    zoom_ratio = config.get("zoom_ratio", 5.0)

    rot_min, rot_max = config["rotation_bounds"]
    scale_min, scale_max = config["scale_bounds"]
    noise_min, noise_max = config["noise_range"]
    speckle_min, speckle_max = config["speckle_range"]
    blur_min, blur_max = config["blur_range"]
    charge_min, charge_max = config["charging_amplitude_range"]
    density_min, density_max = config["pattern_densities"]

    for i in range(num_samples):
        sample_density = rng.uniform(density_min, density_max)
        sample_noise = rng.uniform(noise_min, noise_max)
        sample_speckle = rng.uniform(speckle_min, speckle_max)
        sample_blur = rng.uniform(blur_min, blur_max)
        sample_charge = rng.uniform(charge_min, charge_max) * rng.choice([-1.0, 1.0])
        
        sample_rot = rng.uniform(rot_min, rot_max)
        sample_scale = rng.uniform(scale_min, scale_max)

        search_canvas_w = int(search_w * zoom_ratio)
        search_canvas_h = int(search_h * zoom_ratio)
        
        canvas_w = search_canvas_w + 1000
        canvas_h = search_canvas_h + 1000

        canvas = generate_wafer_canvas(canvas_w, canvas_h, sample_density, rng)

        search_cx = rng.uniform(search_canvas_w / 2.0, canvas_w - search_canvas_w / 2.0)
        search_cy = rng.uniform(search_canvas_h / 2.0, canvas_h - search_canvas_h / 2.0)

        search_crop_canvas = extract_transformed_patch(
            canvas, center=(search_cx, search_cy), size=(search_canvas_w, search_canvas_h), angle_deg=0.0, scale=1.0
        )
        search_img_clean = cv2.resize(search_crop_canvas, (search_w, search_h), interpolation=cv2.INTER_AREA)

        margin_x = ref_w / 2.0
        margin_y = ref_h / 2.0
        max_offset_x = (search_canvas_w / 2.0) - margin_x
        max_offset_y = (search_canvas_h / 2.0) - margin_y

        offset_x_canvas = rng.uniform(-max_offset_x, max_offset_x)
        offset_y_canvas = rng.uniform(-max_offset_y, max_offset_y)

        ref_cx = search_cx + offset_x_canvas
        ref_cy = search_cy + offset_y_canvas

        ref_img_clean = extract_transformed_patch(
            canvas, center=(ref_cx, ref_cy), size=(ref_w, ref_h), angle_deg=sample_rot, scale=sample_scale
        )

        tl_x_canvas = search_cx - (search_canvas_w / 2.0)
        tl_y_canvas = search_cy - (search_canvas_h / 2.0)

        rel_x_canvas = ref_cx - tl_x_canvas
        rel_y_canvas = ref_cy - tl_y_canvas

        true_x = rel_x_canvas / zoom_ratio
        true_y = rel_y_canvas / zoom_ratio

        search_img, search_charge = apply_degradations(
            search_img_clean, noise_std=sample_noise, speckle_std=sample_speckle, blur_sigma=sample_blur, charging_amp=sample_charge, rng=rng
        )
        ref_img, _ = apply_degradations(
            ref_img_clean, noise_std=sample_noise, speckle_std=sample_speckle, blur_sigma=sample_blur, charging_amp=sample_charge * 0.5, rng=rng
        )

        sample_dir = os.path.join(dev_dir, f"sample_{i:03d}")
        os.makedirs(sample_dir, exist_ok=True)

        cv2.imwrite(os.path.join(sample_dir, "search_image.png"), search_img)
        cv2.imwrite(os.path.join(sample_dir, "reference_image.png"), ref_img)

        gt_data = {
            "true_x": float(true_x),
            "true_y": float(true_y),
            "rotation_deg": float(sample_rot),
            "scale_factor": float(sample_scale * zoom_ratio),
            "drift_scale": float(sample_scale),
            "zoom_ratio": float(zoom_ratio),
            "noise_level": float(sample_noise)
        }
        with open(os.path.join(sample_dir, "ground_truth.json"), "w") as f:
            json.dump(gt_data, f, indent=4)

    print("Generation completed successfully.")

def run_evaluation(
    samples_dir: str,
    hybrid_matcher: HybridMatcher,
    k: int = 5,
    alpha: float = 0.5,
    beta: float = 0.5,
    ranking_mode: str = "hybrid"
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluates the hybrid matcher on a directory of samples.
    """
    sample_paths = sorted(glob.glob(os.path.join(samples_dir, "sample_*")))
    
    predictions = {}
    results = []

    for path in sample_paths:
        sample_id = os.path.basename(path)
        ref_path = os.path.join(path, "reference_image.png")
        srch_path = os.path.join(path, "search_image.png")
        gt_path = os.path.join(path, "ground_truth.json")

        reference = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        search = cv2.imread(srch_path, cv2.IMREAD_GRAYSCALE)
        with open(gt_path, "r") as f:
            gt = json.load(f)

        pred_res = hybrid_matcher.match_hybrid(
            reference, search, sample_id=sample_id, k=k, ranking_mode=ranking_mode, alpha=alpha, beta=beta
        )
        
        predictions[sample_id] = pred_res

        # Error math
        final = pred_res["final_prediction"]
        if final["x"] is not None:
            loc_error = np.sqrt((final["x"] - gt["true_x"])**2 + (final["y"] - gt["true_y"])**2)
            
            raw_rot_err = abs(final["rotation"] - gt["rotation_deg"]) % 360.0
            rot_error = raw_rot_err if raw_rot_err <= 180.0 else 360.0 - raw_rot_err
            
            pred_ds = final["scale"] / gt["zoom_ratio"]
            scale_error = abs(pred_ds - gt["drift_scale"])

            loc_ok = loc_error < 3.0
            rot_ok = rot_error < 0.5
            scale_ok = scale_error < 0.02
            all_ok = loc_ok and rot_ok and scale_ok
        else:
            loc_error, rot_error, scale_error = float("nan"), float("nan"), float("nan")
            loc_ok, rot_ok, scale_ok, all_ok = False, False, False, False

        results.append({
            "sample_id": sample_id,
            "loc_error": loc_error,
            "rot_error": rot_error,
            "scale_error": scale_error,
            "all_ok": all_ok,
            "elapsed_classical": pred_res["elapsed_classical_s"],
            "elapsed_dl": pred_res["elapsed_dl_s"],
            "elapsed_total": pred_res["elapsed_total_s"]
        })

    # Summary stats
    loc_errs = [r["loc_error"] for r in results if not np.isnan(r["loc_error"])]
    rot_errs = [r["rot_error"] for r in results if not np.isnan(r["rot_error"])]
    scale_errs = [r["scale_error"] for r in results if not np.isnan(r["scale_error"])]
    total_times = [r["elapsed_total"] for r in results]

    summary = {
        "mean_loc": float(np.mean(loc_errs)) if loc_errs else float("nan"),
        "median_loc": float(np.median(loc_errs)) if loc_errs else float("nan"),
        "max_loc": float(np.max(loc_errs)) if loc_errs else float("nan"),
        "mean_rot": float(np.mean(rot_errs)) if rot_errs else float("nan"),
        "mean_scale": float(np.mean(scale_errs)) if scale_errs else float("nan"),
        "success_rate": sum(1 for r in results if r["all_ok"]) / len(results),
        "mean_time": float(np.mean(total_times)),
        "raw_results": results
    }

    return predictions, summary

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Hybrid Matcher")
    parser.add_argument("--split", choices=["dev", "benchmark"], default="dev", help="Split to evaluate: dev (separate 20 samples) or benchmark (frozen 40 samples)")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick single sample smoke test")
    parser.add_argument("--k", type=int, default=5, help="Number of candidates")
    parser.add_argument("--alpha", type=float, default=0.5, help="Classical weight")
    parser.add_argument("--beta", type=float, default=0.5, help="DL weight")
    args = parser.parse_args()

    config = load_config("config.yaml")
    
    # Initialize Hybrid Matcher
    hybrid_matcher = HybridMatcher(
        zoom_ratio=float(config.get("zoom_ratio", 5.0)),
        rot_range_deg=tuple(config.get("rotation_bounds", [-3.0, 3.0])),
        scale_range=tuple(config.get("scale_bounds", [0.97, 1.03])),
        checkpoint_path="models/dl_matcher/best_model.pth"
    )

    if args.split == "dev":
        # Generate dev samples if not present
        generate_dev_samples(num_samples=20, dev_dir="data/dev_samples")
        samples_dir = "data/dev_samples"
    else:
        samples_dir = "data" # frozen benchmark

    if args.smoke_test:
        print("\n--- Running Smoke Test ---")
        sample_path = glob.glob(os.path.join(samples_dir, "sample_*"))[0]
        sample_id = os.path.basename(sample_path)
        print(f"Testing on sample: {sample_id}")
        
        ref = cv2.imread(os.path.join(sample_path, "reference_image.png"), cv2.IMREAD_GRAYSCALE)
        srch = cv2.imread(os.path.join(sample_path, "search_image.png"), cv2.IMREAD_GRAYSCALE)
        
        res = hybrid_matcher.match_hybrid(ref, srch, sample_id=sample_id, k=args.k, ranking_mode="hybrid", alpha=args.alpha, beta=args.beta)
        print(f"Top candidate: {res['final_prediction']}")
        print(f"Candidates count: {len(res['candidates'])}")
        print("Smoke test successfully completed.")
        return

    # Evaluation systems
    print(f"\nEvaluating systems on {args.split} split...")
    
    # 1. Classical Baseline
    print("Running Classical Baseline...")
    _, class_sum = run_evaluation(samples_dir, hybrid_matcher, k=args.k, ranking_mode="classical")

    # 2. DL Reranking
    print("Running DL Reranking...")
    _, dl_sum = run_evaluation(samples_dir, hybrid_matcher, k=args.k, ranking_mode="dl")

    # 3. Hybrid Score Fusion
    print(f"Running Hybrid Fusion (alpha={args.alpha}, beta={args.beta})...")
    hybrid_preds, hybrid_sum = run_evaluation(samples_dir, hybrid_matcher, k=args.k, alpha=args.alpha, beta=args.beta, ranking_mode="hybrid")

    # Output predictions for evaluation saving
    pred_save_path = f"data/predictions_hybrid_{args.split}.json"
    with open(pred_save_path, "w") as f:
        json.dump(hybrid_preds, f, indent=4)
    print(f"Hybrid predictions saved to {pred_save_path}")

    # Print summary comparative table
    print("\n" + "="*80)
    print(f"COMPARATIVE EVALUATION SUMMARY ({args.split.upper()} SPLIT)")
    print("="*80)
    print(f"{'Metric':<25} | {'Classical Only':<15} | {'DL Only':<15} | {'Hybrid Fusion':<15}")
    print("-"*80)
    print(f"{'Mean Loc Error (px)':<25} | {class_sum['mean_loc']:<15.4f} | {dl_sum['mean_loc']:<15.4f} | {hybrid_sum['mean_loc']:<15.4f}")
    print(f"{'Median Loc Error (px)':<25} | {class_sum['median_loc']:<15.4f} | {dl_sum['median_loc']:<15.4f} | {hybrid_sum['median_loc']:<15.4f}")
    print(f"{'Max Loc Error (px)':<25} | {class_sum['max_loc']:<15.4f} | {dl_sum['max_loc']:<15.4f} | {hybrid_sum['max_loc']:<15.4f}")
    print(f"{'Mean Rot Error (°)':<25} | {class_sum['mean_rot']:<15.4f} | {dl_sum['mean_rot']:<15.4f} | {hybrid_sum['mean_rot']:<15.4f}")
    print(f"{'Mean Scale Error':<25} | {class_sum['mean_scale']:<15.5f} | {dl_sum['mean_scale']:<15.5f} | {hybrid_sum['mean_scale']:<15.5f}")
    print(f"{'Success Rate (%)':<25} | {class_sum['success_rate']*100:<15.1f} | {dl_sum['success_rate']*100:<15.1f} | {hybrid_sum['success_rate']*100:<15.1f}")
    print(f"{'Mean Inference Time (s)':<25} | {class_sum['mean_time']:<15.4f} | {dl_sum['mean_time']:<15.4f} | {hybrid_sum['mean_time']:<15.4f}")
    print("="*80)

    # 4. Candidate Ranking Analysis
    # Load GT values
    gt_map = {}
    for p in glob.glob(os.path.join(samples_dir, "sample_*", "ground_truth.json")):
        sid = os.path.basename(os.path.dirname(p))
        with open(p, "r") as f:
            gt_map[sid] = json.load(f)

    # Count how often classical Top-1 is correct
    # Count how often correct candidate appears in Top-K
    class_top1_ok = 0
    correct_in_top_k = 0
    dl_improved_rank = 0
    dl_worse_rank = 0
    hybrid_improved = 0

    change_list = []

    for sid in gt_map:
        pred_item = hybrid_preds[sid]
        gt_item = gt_map[sid]
        
        # Check candidates
        cands = pred_item["candidates"]
        
        # Determine success of each candidate
        correct_ranks = []
        for c in cands:
            loc_err = np.sqrt((c["x"] - gt_item["true_x"])**2 + (c["y"] - gt_item["true_y"])**2)
            raw_rot = abs(c["rotation"] - gt_item["rotation_deg"]) % 360.0
            rot_err = raw_rot if raw_rot <= 180.0 else 360.0 - raw_rot
            pred_ds = c["scale"] / gt_item["zoom_ratio"]
            scale_err = abs(pred_ds - gt_item["drift_scale"])
            
            is_ok = loc_err < 3.0 and rot_err < 0.5 and scale_err < 0.02
            if is_ok:
                correct_ranks.append(c["rank_before_dl"])

        # Top-1 classical success check
        if 1 in correct_ranks:
            class_top1_ok += 1
        # Success in any rank check
        if len(correct_ranks) > 0:
            correct_in_top_k += 1

        # Check candidate ranks after reranking
        # We find the final selected rank
        sel_rank_hybrid = pred_item["final_prediction"]["selected_candidate_rank"]
        
        # DL only best rank
        dl_best_cand = sorted(cands, key=lambda x: x["match_probability"], reverse=True)[0]
        sel_rank_dl = dl_best_cand["rank_before_dl"]

        # Did it improve over classical top-1?
        # If classical top-1 was wrong and selected rank was correct: improvement
        # If classical top-1 was correct and selected rank was wrong: worse
        if 1 not in correct_ranks:
            # Classical was wrong anyway
            pass
        else:
            # Classical top-1 was correct. Did DL or hybrid select a wrong candidate?
            if sel_rank_dl not in correct_ranks:
                dl_worse_rank += 1
            if sel_rank_hybrid not in correct_ranks:
                pass # hybrid worse

        # If classical was wrong, did DL or hybrid correct it?
        if 1 not in correct_ranks and len(correct_ranks) > 0:
            if sel_rank_dl in correct_ranks:
                dl_improved_rank += 1
            if sel_rank_hybrid in correct_ranks:
                hybrid_improved += 1

        # Record changes in selected candidate
        if sel_rank_hybrid != 1:
            # Selected candidate changed!
            # Find the details
            # Find classical details
            c_cand = [c for c in cands if c["rank_before_dl"] == 1][0]
            # Find hybrid details
            h_cand = [c for c in cands if c["rank_before_dl"] == sel_rank_hybrid][0]
            
            # Find DL rank of selected hybrid candidate
            dl_sorted = sorted(cands, key=lambda x: x["match_probability"], reverse=True)
            dl_rank_of_h = [i+1 for i, x in enumerate(dl_sorted) if x["rank_before_dl"] == sel_rank_hybrid][0]
            dl_rank_of_c = [i+1 for i, x in enumerate(dl_sorted) if x["rank_before_dl"] == 1][0]

            change_list.append({
                "sample_id": sid,
                "classical_rank": 1,
                "dl_rank_of_selected": dl_rank_of_h,
                "hybrid_rank_of_selected": 1, # since it's now top-1 after hybrid sorting
                "classical_score": c_cand["classical_score"],
                "selected_classical_score": h_cand["classical_score"],
                "classical_prob": c_cand["match_probability"],
                "selected_prob": h_cand["match_probability"],
                "final_selected_score": h_cand["hybrid_score"]
            })

    total_samples = len(gt_map)
    print("\n--- CANDIDATE RANKING ANALYSIS ---")
    print(f"Classical Top-1 Correct:           {class_top1_ok} / {total_samples} ({100*class_top1_ok/total_samples:.1f}%)")
    print(f"Correct Candidate in Top-K:        {correct_in_top_k} / {total_samples} ({100*correct_in_top_k/total_samples:.1f}%)")
    print(f"DL Reranking Improved Wrong Rank:  {dl_improved_rank} samples")
    print(f"DL Reranking Degraded Correct Rank: {dl_worse_rank} samples")
    print(f"Hybrid Fusion Corrected Wrong Rank: {hybrid_improved} samples")
    
    print("\n--- SAMPLES WITH CHANGED SELECTION ---")
    if not change_list:
        print("No samples changed their selected candidate.")
    else:
        print(f"{'Sample':<12} | {'ClassScore (C vs H)':<20} | {'DL Prob (C vs H)':<20} | {'FinalScore (H)':<15}")
        print("-"*80)
        for chg in change_list:
            print(f"{chg['sample_id']:<12} | {chg['classical_score']:.3f} vs {chg['selected_classical_score']:.3f} | {chg['classical_prob']:.3f} vs {chg['selected_prob']:.3f} | {chg['final_selected_score']:.4f}")

if __name__ == "__main__":
    main()
