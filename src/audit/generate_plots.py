"""
Generates the final report visualizations (Figures 1-5) and saves them in high resolution.
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("Generating report visualizations...")
    fig_dir = "reports/final_results/figures"
    os.makedirs(fig_dir, exist_ok=True)
    
    # Set default style parameters
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 14
    })
    
    # Load metrics from JSON files
    with open("reports/final_freeze/benchmark_metrics.json", "r") as f:
        bm = json.load(f)
    with open("results/phase7_robustness/aggregate_metrics.json", "r") as f:
        rm = json.load(f)
        
    bm_locs = [r["loc_error"] for r in bm["raw_results"]]
    
    # Compute robustness location errors dynamically
    with open("results/phase7_robustness/predictions.json", "r") as f:
        preds = json.load(f)
    rm_locs = []
    for i in range(200):
        sample_id = f"sample_{i:03d}"
        gt_path = f"data/robustness_samples/{sample_id}/ground_truth.json"
        with open(gt_path, "r") as f:
            gt = json.load(f)
        pred = preds[sample_id]
        loc_err = float(np.sqrt((pred["final_prediction"]["x"] - gt["true_x"])**2 + (pred["final_prediction"]["y"] - gt["true_y"])**2))
        rm_locs.append(loc_err)

    # ----------------------------------------------------
    # Figure 1: Benchmark vs Robustness Success Rate
    # ----------------------------------------------------
    plt.figure(figsize=(6, 5))
    categories = ['Frozen Benchmark\n(40 samples)', 'Robustness Set\n(200 samples)']
    rates = [bm["success_rate"] * 100.0, rm["success_rate"] * 100.0]
    
    bars = plt.bar(categories, rates, color=['#1f77b4', '#2ca02c'], width=0.5, edgecolor='black', alpha=0.85)
    plt.ylabel('Success Rate (%)')
    plt.title('Figure 1: Success Rate Comparison')
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom', weight='bold')
                     
    plt.tight_layout()
    fig1_path = os.path.join(fig_dir, "figure1_success_rate.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print("Figure 1 saved.")

    # ----------------------------------------------------
    # Figure 2: Localization Error Distribution (Log Scale)
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Combine data for plotting
    data = [bm_locs, rm_locs]
    
    # Plot boxplot on a log-scale to show outliers clearly
    bp = ax.boxplot(data, patch_artist=True, widths=0.4,
                    boxprops=dict(facecolor='#e2e2e2', color='black', alpha=0.7),
                    medianprops=dict(color='red', linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='red', markersize=6, linestyle='none', markeredgecolor='black'))
                    
    # Annotate metrics
    ax.set_yscale('log')
    ax.set_xticklabels(['Frozen Benchmark\n(40 samples)', 'Robustness Set\n(200 samples)'])
    ax.set_ylabel('Localization Error (pixels) - Log Scale')
    ax.set_title('Figure 2: Localization Error Distribution')
    ax.grid(axis='y', which='both', linestyle='--', alpha=0.3)
    
    # Text annotations for median, p95, and max
    bm_median = np.median(bm_locs)
    bm_max = np.max(bm_locs)
    
    rm_median = rm["median_loc"]
    rm_p95 = rm["p95_loc"]
    rm_max = rm["max_loc"]
    
    # Benchmark stats text
    ax.text(0.65, bm_median, f"Med: {bm_median:.3f} px\nMax: {bm_max:.3f} px", 
            color='blue', fontsize=10, ha='center', va='bottom', weight='bold')
            
    # Robustness stats text
    ax.text(2.35, rm_median, f"Med: {rm_median:.3f} px\nP95: {rm_p95:.3f} px\nMax: {rm_max:.3f} px", 
            color='darkgreen', fontsize=10, ha='center', va='bottom', weight='bold')
            
    plt.tight_layout()
    fig2_path = os.path.join(fig_dir, "figure2_loc_error_dist.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print("Figure 2 saved.")

    # ----------------------------------------------------
    # Figure 3: Robustness by Noise Level
    # ----------------------------------------------------
    plt.figure(figsize=(7, 5))
    noise_bins = ['Low Noise\n(<= 0.02)', 'Medium Noise\n(0.02 - 0.04)', 'High Noise\n(0.04 - 0.06)']
    noise_rates = [100.0, 79/81 * 100.0, 85/88 * 100.0]
    
    bars = plt.bar(noise_bins, noise_rates, color='#9467bd', width=0.5, edgecolor='black', alpha=0.8)
    plt.ylabel('Success Rate (%)')
    plt.title('Figure 3: Robustness Success Rate by Noise Level')
    plt.ylim(90, 102) # Focused zoom to see differences
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', weight='bold')
                     
    plt.tight_layout()
    fig3_path = os.path.join(fig_dir, "figure3_noise_breakdown.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print("Figure 3 saved.")

    # ----------------------------------------------------
    # Figure 4: Ablation Comparison (Success Rate)
    # ----------------------------------------------------
    plt.figure(figsize=(9, 5))
    systems = [
        'Original Phase 3\nClassical',
        'Classical +\nRefinement',
        'DL Matcher V1\nReranking',
        'Hybrid\nFusion',
        'Final Frozen\nSystem'
    ]
    rates = [77.5, 97.5, 27.5, 72.5, 97.5]
    
    bars = plt.bar(systems, rates, color=['#bcbd22', '#17becf', '#d62728', '#ff7f0e', '#1f77b4'], width=0.5, edgecolor='black', alpha=0.85)
    plt.ylabel('Success Rate (%)')
    plt.title('Figure 4: Ablation Success Rate Comparison')
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', weight='bold')
                     
    plt.tight_layout()
    fig4_path = os.path.join(fig_dir, "figure4_ablation_comparison.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print("Figure 4 saved.")

    # ----------------------------------------------------
    # Figure 5: Inference Time Comparison
    # ----------------------------------------------------
    plt.figure(figsize=(9, 5))
    times = [0.6153, 0.6267, 0.6153, 0.6153, 0.3666]
    
    bars = plt.bar(systems, times, color=['#7f7f7f', '#c7c7c7', '#e377c2', '#8c564b', '#2ca02c'], width=0.5, edgecolor='black', alpha=0.85)
    plt.ylabel('Average Inference Time (seconds)')
    plt.title('Figure 5: Average Inference Time per Sample')
    plt.ylim(0, 0.8)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.4f} s',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', weight='bold')
                     
    plt.tight_layout()
    fig5_path = os.path.join(fig_dir, "figure5_inference_time.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print("Figure 5 saved.")
    print("All visualizations generated successfully.")

if __name__ == "__main__":
    main()
