"""
Generates success, failure, and gallery alignment visualizations from frozen predictions.
"""
import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_transformed_corners(x: float, y: float, rot_deg: float, scale: float, target_size: int = 256) -> np.ndarray:
    theta = np.radians(rot_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    half_size = target_size / 2.0
    # Corner offsets relative to center (128, 128)
    local_corners = np.array([
        [-half_size, -half_size],
        [half_size, -half_size],
        [half_size, half_size],
        [-half_size, half_size],
        [-half_size, -half_size] # Close the polygon
    ])
    
    transformed = []
    for du, dv in local_corners:
        px = x + (du * cos_t - dv * sin_t) / scale
        py = y + (du * sin_t + dv * cos_t) / scale
        transformed.append([px, py])
    return np.array(transformed)

def generate_panel_plot(
    sample_id: str,
    dataset_name: str,
    ref_img: np.ndarray,
    search_img: np.ndarray,
    pred_x: float,
    pred_y: float,
    pred_rot: float,
    pred_scale: float,
    true_x: float,
    true_y: float,
    true_rot: float,
    true_scale: float,
    loc_err: float,
    rot_err: float,
    scale_err: float,
    failed_criterion: str = None,
    is_tracking_loss: bool = False,
    is_failure_case: bool = False
) -> plt.Figure:
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    # Panel A: Reference
    axs[0, 0].imshow(ref_img, cmap='gray')
    axs[0, 0].set_title("Panel A: Reference Patch (256x256)")
    axs[0, 0].axis('off')
    
    # Panel B: Search Image
    axs[0, 1].imshow(search_img, cmap='gray')
    axs[0, 1].plot(pred_x, pred_y, 'ro', label='Predicted Center', markersize=6)
    axs[0, 1].plot(true_x, true_y, 'gx', label='Ground Truth Center', markersize=8)
    axs[0, 1].set_title("Panel B: Search Image (512x512)")
    axs[0, 1].legend(loc='upper right')
    axs[0, 1].axis('off')
    
    # Panel C: Local Alignment (Zoom around true center)
    zoom_size = 64
    x_min = max(0, int(true_x - zoom_size))
    x_max = min(search_img.shape[1], int(true_x + zoom_size))
    y_min = max(0, int(true_y - zoom_size))
    y_max = min(search_img.shape[0], int(true_y + zoom_size))
    
    zoom_patch = search_img[y_min:y_max, x_min:x_max]
    axs[1, 0].imshow(zoom_patch, cmap='gray', extent=[x_min, x_max, y_max, y_min])
    
    # Get corners of predicted and true patches in search space
    pred_corners = get_transformed_corners(pred_x, pred_y, pred_rot, pred_scale)
    true_corners = get_transformed_corners(true_x, true_y, true_rot, true_scale)
    
    axs[1, 0].plot(pred_corners[:, 0], pred_corners[:, 1], 'r-', label='Pred Boundary', linewidth=1.5)
    axs[1, 0].plot(true_corners[:, 0], true_corners[:, 1], 'g--', label='True Boundary', linewidth=1.5)
    axs[1, 0].plot(pred_x, pred_y, 'ro', markersize=4)
    axs[1, 0].plot(true_x, true_y, 'gx', markersize=5)
    axs[1, 0].set_xlim(x_min, x_max)
    axs[1, 0].set_ylim(y_max, y_min) # Inverted for correct image orientation
    axs[1, 0].set_title("Panel C: Local Zoom & Boundaries")
    axs[1, 0].legend(loc='lower left', fontsize=8)
    
    # Panel D: Error Summary
    axs[1, 1].axis('off')
    title_color = 'red' if is_failure_case else 'black'
    header = f"Panel D: Summary ({sample_id})"
    if is_failure_case:
        header += " [FAILURE]"
    axs[1, 1].text(0.05, 0.9, header, fontsize=12, color=title_color, weight='bold')
    
    summary_text = (
        f"Dataset: {dataset_name}\n\n"
        f"Predictions:\n"
        f"  X, Y: {pred_x:.4f}, {pred_y:.4f}\n"
        f"  Rotation: {pred_rot:.4f}°\n"
        f"  Scale: {pred_scale:.4f}\n\n"
        f"Ground Truth:\n"
        f"  X, Y: {true_x:.4f}, {true_y:.4f}\n"
        f"  Rotation: {true_rot:.4f}°\n"
        f"  Scale: {true_scale:.4f}\n\n"
        f"Error Metrics:\n"
        f"  Localization Error: {loc_err:.4f} px (Gate: < 3.0)\n"
        f"  Rotation Error: {rot_err:.4f}° (Gate: < 0.5)\n"
        f"  Scale Error: {scale_err:.5f} (Gate: < 0.02)\n"
    )
    
    if is_failure_case:
        summary_text += f"\nFAILED CRITERION: {failed_criterion.upper()}"
        if is_tracking_loss:
            summary_text += "\n*** TRACKING LOSS DETECTED ***"
            
    axs[1, 1].text(0.05, 0.1, summary_text, fontsize=9.5, family='monospace', va='top')
    
    plt.tight_layout()
    return fig

def main():
    print("Initializing alignment visualization creation...")
    
    # 1. Load predictions
    with open("data/final_predictions_benchmark.json", "r") as f:
        bm_preds = json.load(f)
    with open("results/phase7_robustness/predictions.json", "r") as f:
        rm_preds = json.load(f)
        
    traceability = []
    
    # Setup folders
    success_dir = "reports/final_results/alignment_examples/success"
    os.makedirs(success_dir, exist_ok=True)
    
    benchmark_failure_dir = "reports/final_results/alignment_examples/frozen_failure/sample_021"
    os.makedirs(benchmark_failure_dir, exist_ok=True)
    
    robustness_failure_dir = "reports/final_results/alignment_examples/robustness_failures"
    os.makedirs(robustness_failure_dir, exist_ok=True)
    
    figures_dir = "reports/final_results/figures"
    os.makedirs(figures_dir, exist_ok=True)

    # ----------------------------------------------------
    # SUCCESS CASES (3 representative samples)
    # ----------------------------------------------------
    success_samples = [
        ("sample_000", "Frozen Benchmark", "data/sample_000", bm_preds["sample_000"]),
        ("sample_010", "Robustness Set", "data/robustness_samples/sample_010", rm_preds["sample_010"]),
        ("sample_020", "Robustness Set", "data/robustness_samples/sample_020", rm_preds["sample_020"]),
        ("sample_030", "Robustness Set", "data/robustness_samples/sample_030", rm_preds["sample_030"])
    ]
    
    success_figs_to_gallery = []
    
    for sample_id, dataset_name, data_path, pred in success_samples:
        ref_path = os.path.join(data_path, "reference_image.png")
        srch_path = os.path.join(data_path, "search_image.png")
        gt_path = os.path.join(data_path, "ground_truth.json")
        
        ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        srch = cv2.imread(srch_path, cv2.IMREAD_GRAYSCALE)
        with open(gt_path, "r") as f:
            gt = json.load(f)
            
        # Parse prediction values
        if "final_prediction" in pred:
            px, py = pred["final_prediction"]["x"], pred["final_prediction"]["y"]
            prot, pscale = pred["final_prediction"]["rotation"], pred["final_prediction"]["scale"]
        else:
            px, py = pred["predicted_x"], pred["predicted_y"]
            prot, pscale = pred["predicted_rotation"], pred["predicted_scale"]
            
        # Errors
        loc_err = float(np.sqrt((px - gt["true_x"])**2 + (py - gt["true_y"])**2))
        raw_rot_err = abs(prot - gt["rotation_deg"]) % 360.0
        rot_err = float(raw_rot_err if raw_rot_err <= 180.0 else 360.0 - raw_rot_err)
        scale_err = float(abs((pscale / gt["zoom_ratio"]) - gt["drift_scale"]))
        
        fig = generate_panel_plot(
            sample_id, dataset_name, ref, srch,
            px, py, prot, pscale,
            gt["true_x"], gt["true_y"], gt["rotation_deg"], gt["drift_scale"] * gt["zoom_ratio"],
            loc_err, rot_err, scale_err
        )
        
        out_file = os.path.join(success_dir, f"{sample_id}_alignment.png")
        fig.savefig(out_file, dpi=300)
        plt.close(fig)
        success_figs_to_gallery.append((sample_id, loc_err, rot_err, scale_err, out_file))
        
        traceability.append({
            "sample_id": sample_id,
            "dataset": dataset_name,
            "prediction_source": "data/final_predictions_benchmark.json" if dataset_name == "Frozen Benchmark" else "results/phase7_robustness/predictions.json",
            "ground_truth_source": gt_path,
            "metadata_source": gt_path,
            "output_file": out_file
        })
        print(f"Success visualization generated for {sample_id}.")

    # ----------------------------------------------------
    # FROZEN BENCHMARK FAILURE: sample_021
    # ----------------------------------------------------
    print("Generating frozen benchmark failure visualization for sample_021...")
    sample_id = "sample_021"
    dataset_name = "Frozen Benchmark"
    data_path = "data/sample_021"
    pred = bm_preds["sample_021"]
    
    ref_path = os.path.join(data_path, "reference_image.png")
    srch_path = os.path.join(data_path, "search_image.png")
    gt_path = os.path.join(data_path, "ground_truth.json")
    
    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    srch = cv2.imread(srch_path, cv2.IMREAD_GRAYSCALE)
    with open(gt_path, "r") as f:
        gt = json.load(f)
        
    px, py = pred["predicted_x"], pred["predicted_y"]
    prot, pscale = pred["predicted_rotation"], pred["predicted_scale"]
    
    loc_err = float(np.sqrt((px - gt["true_x"])**2 + (py - gt["true_y"])**2))
    raw_rot_err = abs(prot - gt["rotation_deg"]) % 360.0
    rot_err = float(raw_rot_err if raw_rot_err <= 180.0 else 360.0 - raw_rot_err)
    scale_err = float(abs((pscale / gt["zoom_ratio"]) - gt["drift_scale"]))
    
    # Detailed failure plot for sample_021
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    # panel 1: Search image with labels
    axs[0, 0].imshow(srch, cmap='gray')
    axs[0, 0].plot(px, py, 'ro', label='Predicted Center')
    axs[0, 0].plot(gt["true_x"], gt["true_y"], 'gx', label='True Center')
    axs[0, 0].set_title("1. Search Image")
    axs[0, 0].legend()
    axs[0, 0].axis('off')
    
    # panel 2: Zoomed area highlighting 0.88 px translation offset
    zoom_size = 16
    x_min = int(gt["true_x"] - zoom_size)
    x_max = int(gt["true_x"] + zoom_size)
    y_min = int(gt["true_y"] - zoom_size)
    y_max = int(gt["true_y"] + zoom_size)
    
    axs[0, 1].imshow(srch[y_min:y_max, x_min:x_max], cmap='gray', extent=[x_min, x_max, y_max, y_min])
    axs[0, 1].plot(px, py, 'ro', markersize=8, label='Predicted')
    axs[0, 1].plot(gt["true_x"], gt["true_y"], 'gx', markersize=10, label='Ground Truth')
    axs[0, 1].set_title(f"2. Local Zoom (Offset: {loc_err:.4f} px)")
    axs[0, 1].legend()
    
    # panel 3: Rotation comparison
    axs[1, 0].axis('off')
    rot_text = (
        f"3. Rotation Comparison\n\n"
        f"Predicted Rotation: {prot:.4f}°\n"
        f"Ground Truth Rotation: {gt['rotation_deg']:.4f}°\n"
        f"Rotation Error: {rot_err:.4f}°\n"
        f"Success Gate Threshold: < 0.5°\n"
        f"Status: FAILED\n\n"
        f"Intermediate refinement trace unavailable from saved outputs."
    )
    axs[1, 0].text(0.05, 0.8, rot_text, fontsize=10.5, family='monospace', va='top')
    
    # panel 4: Causal explanation
    axs[1, 1].axis('off')
    explanation_text = (
        "4. Failure Explanation\n\n"
        "Residual translation error was present while\n"
        "X/Y coordinates were held fixed during rotation\n"
        "and scale refinement. The measured NCC objective\n"
        "selected a biased rotation under this\n"
        "fixed-center condition."
    )
    axs[1, 1].text(0.05, 0.8, explanation_text, fontsize=10.5, family='sans-serif', va='top', bbox=dict(facecolor='#ffcccc', alpha=0.5))
    
    plt.tight_layout()
    out_file_021 = os.path.join(benchmark_failure_dir, "sample_021_failure.png")
    fig.savefig(out_file_021, dpi=300)
    plt.close(fig)
    
    traceability.append({
        "sample_id": sample_id,
        "dataset": dataset_name,
        "prediction_source": "data/final_predictions_benchmark.json",
        "ground_truth_source": gt_path,
        "metadata_source": gt_path,
        "output_file": out_file_021
    })

    # ----------------------------------------------------
    # ROBUSTNESS FAILURES (5 known samples)
    # ----------------------------------------------------
    robustness_fails = [
        ("sample_006", "localization", "Extreme noise/degradation", True),
        ("sample_054", "scale", "Ambiguous correlation peaks", False),
        ("sample_092", "localization", "Extreme noise/degradation", True),
        ("sample_121", "localization", "Ambiguous correlation peaks", True),
        ("sample_166", "localization", "Coupling of translation error into rotation refinement", True)
    ]
    
    robustness_figs_to_gallery = []
    
    for sample_id, failed_criterion, category, is_tracking_loss in robustness_fails:
        data_path = f"data/robustness_samples/{sample_id}"
        pred = rm_preds[sample_id]
        
        ref_path = os.path.join(data_path, "reference_image.png")
        srch_path = os.path.join(data_path, "search_image.png")
        gt_path = os.path.join(data_path, "ground_truth.json")
        
        ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        srch = cv2.imread(srch_path, cv2.IMREAD_GRAYSCALE)
        with open(gt_path, "r") as f:
            gt = json.load(f)
            
        px, py = pred["final_prediction"]["x"], pred["final_prediction"]["y"]
        prot, pscale = pred["final_prediction"]["rotation"], pred["final_prediction"]["scale"]
        
        loc_err = float(np.sqrt((px - gt["true_x"])**2 + (py - gt["true_y"])**2))
        raw_rot_err = abs(prot - gt["rotation_deg"]) % 360.0
        rot_err = float(raw_rot_err if raw_rot_err <= 180.0 else 360.0 - raw_rot_err)
        scale_err = float(abs((pscale / gt["zoom_ratio"]) - gt["drift_scale"]))
        
        # Override failed criteria description if multiple gates failed
        # (e.g. sample_166 or sample_092)
        gates_list = []
        if loc_err >= 3.0:
            gates_list.append("localization")
        if rot_err >= 0.5:
            gates_list.append("rotation")
        if scale_err >= 0.02:
            gates_list.append("scale")
        failed_gates_str = ", ".join(gates_list)
        
        fig = generate_panel_plot(
            sample_id, "Robustness Set", ref, srch,
            px, py, prot, pscale,
            gt["true_x"], gt["true_y"], gt["rotation_deg"], gt["drift_scale"] * gt["zoom_ratio"],
            loc_err, rot_err, scale_err,
            failed_criterion=failed_gates_str,
            is_tracking_loss=is_tracking_loss,
            is_failure_case=True
        )
        
        out_file = os.path.join(robustness_failure_dir, f"{sample_id}_failure.png")
        fig.savefig(out_file, dpi=300)
        plt.close(fig)
        robustness_figs_to_gallery.append((sample_id, loc_err, rot_err, scale_err, failed_gates_str, out_file))
        
        traceability.append({
            "sample_id": sample_id,
            "dataset": "Robustness Set",
            "prediction_source": "results/phase7_robustness/predictions.json",
            "ground_truth_source": gt_path,
            "metadata_source": gt_path,
            "output_file": out_file
        })
        print(f"Robustness failure visualization generated for {sample_id}.")

    # ----------------------------------------------------
    # COMPOSITE FAILURE GALLERY
    # ----------------------------------------------------
    print("Generating composite failure gallery...")
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Frozen Benchmark Failure (sample_021)
    axs[0, 0].imshow(srch, cmap='gray') # reuse sample_021's images
    axs[0, 0].plot(bm_preds["sample_021"]["predicted_x"], bm_preds["sample_021"]["predicted_y"], 'ro', markersize=6)
    with open("data/sample_021/ground_truth.json", "r") as f:
        sample_021_gt = json.load(f)
    axs[0, 0].plot(sample_021_gt["true_x"], sample_021_gt["true_y"], 'gx', markersize=8)
    axs[0, 0].set_title("Frozen Benchmark: sample_021\n(Loc Err: 0.881 px, Rot Err: 0.543°)")
    axs[0, 0].axis('off')
    
    # Plot 2-6: Robustness failures
    fail_idx = 0
    for r in range(2):
        for c in range(3):
            if r == 0 and c == 0:
                continue # Already benchmark sample
            
            sample_id, l_e, r_e, s_e, gates_str, f_path = robustness_figs_to_gallery[fail_idx]
            
            # Load search image for this sample
            srch_f = cv2.imread(f"data/robustness_samples/{sample_id}/search_image.png", cv2.IMREAD_GRAYSCALE)
            with open(f"data/robustness_samples/{sample_id}/ground_truth.json", "r") as f:
                f_gt = json.load(f)
            pred_f = rm_preds[sample_id]
            px_f, py_f = pred_f["final_prediction"]["x"], pred_f["final_prediction"]["y"]
            
            axs[r, c].imshow(srch_f, cmap='gray')
            axs[r, c].plot(px_f, py_f, 'ro', markersize=6)
            axs[r, c].plot(f_gt["true_x"], f_gt["true_y"], 'gx', markersize=8)
            
            label = f"Robustness: {sample_id}\n(Loc: {l_e:.2f}px, Rot: {r_e:.2f}°, Scale: {s_e:.4f})"
            if l_e > 50.0:
                label += "\n[TRACKING LOSS]"
            axs[r, c].set_title(label, fontsize=10)
            axs[r, c].axis('off')
            fail_idx += 1
            
    plt.suptitle("Summary Failure Gallery (Benchmark vs Robustness Set)", fontsize=16, weight='bold')
    plt.tight_layout()
    gallery_fail_path = os.path.join(figures_dir, "failure_gallery.png")
    fig.savefig(gallery_fail_path, dpi=300)
    plt.close(fig)
    print("Composite failure gallery written.")

    # ----------------------------------------------------
    # COMPOSITE SUCCESS GALLERY
    # ----------------------------------------------------
    print("Generating composite success gallery...")
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Use 3 robustness success samples
    for idx, (sample_id, l_e, r_e, s_e, f_path) in enumerate(success_figs_to_gallery[1:]): # skip the benchmark sample
        srch_s = cv2.imread(f"data/robustness_samples/{sample_id}/search_image.png", cv2.IMREAD_GRAYSCALE)
        with open(f"data/robustness_samples/{sample_id}/ground_truth.json", "r") as f:
            s_gt = json.load(f)
        pred_s = rm_preds[sample_id]
        px_s, py_s = pred_s["final_prediction"]["x"], pred_s["final_prediction"]["y"]
        
        axs[idx].imshow(srch_s, cmap='gray')
        axs[idx].plot(px_s, py_s, 'ro', label='Pred')
        axs[idx].plot(s_gt["true_x"], s_gt["true_y"], 'gx', label='True')
        axs[idx].set_title(f"Success Sample: {sample_id}\n(Loc: {l_e:.3f} px, Rot: {r_e:.3f}°, Scale: {s_e:.4f})", fontsize=11)
        axs[idx].axis('off')
        
    plt.suptitle("Success Gallery (Representative Robustness Alignments)", fontsize=15, weight='bold')
    plt.tight_layout()
    gallery_success_path = os.path.join(figures_dir, "success_gallery.png")
    fig.savefig(gallery_success_path, dpi=300)
    plt.close(fig)
    print("Composite success gallery written.")
    
    # 5. Write visualization_traceability.json
    with open("reports/final_results/visualization_traceability.json", "w") as f:
        json.dump(traceability, f, indent=4)
    print("visualization_traceability.json written successfully.")

if __name__ == "__main__":
    main()
