import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, List

from src.ml.model import SiameseMatcher
from src.ml_v2.model_v2 import SiameseMatcherV2
from src.ml_v2.train_matcher_v2 import WaferTripletDataset

def compute_metrics(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    desc_score_indices = np.argsort(y_prob)[::-1]
    y_prob = y_prob[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    distinct_value_indices = np.where(np.diff(y_prob))[0]
    threshold_idxs = np.r_[distinct_value_indices, len(y_true) - 1]
    
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    
    if tps[-1] == 0 or fps[-1] == 0:
        roc_auc = 0.5
    else:
        fpr = fps / fps[-1]
        tpr = tps / tps[-1]
        roc_auc = float(np.sum(0.5 * (tpr[:-1] + tpr[1:]) * np.diff(fpr)))
        
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc)
    }

def run_model_evaluation(model, loader, device, model_name: str) -> Dict[str, Any]:
    model.eval()
    
    all_y_true = []
    all_y_prob = []
    
    # Ranking metrics accumulation
    total_triplets = 0
    correct_rankings = 0
    pos_similarities = []
    neg_similarities = []
    margins = []
    
    # Categorized ranking accumulation
    categories = ["random", "nearby", "repeated", "wrong_geom", "classical_hard"]
    cat_correct = {c: 0 for c in categories}
    cat_total = {c: 0 for c in categories}
    
    with torch.no_grad():
        for ref, pos, neg, metas in loader:
            ref, pos, neg = ref.to(device), pos.to(device), neg.to(device)
            
            # Forward pass
            prob_pos, sim_pos = model(ref, pos)
            prob_neg, sim_neg = model(ref, neg)
            
            prob_pos = prob_pos.cpu().numpy().flatten()
            prob_neg = prob_neg.cpu().numpy().flatten()
            sim_pos = sim_pos.cpu().numpy().flatten()
            sim_neg = sim_neg.cpu().numpy().flatten()
            
            # Accumulate classification metrics
            for p_p, p_n in zip(prob_pos, prob_neg):
                all_y_true.extend([1, 0])
                all_y_prob.extend([p_p, p_n])
                
            # Accumulate ranking metrics
            batch_size = ref.size(0)
            for i in range(batch_size):
                s_p = sim_pos[i]
                s_n = sim_neg[i]
                neg_type = metas["neg_type"][i]
                
                is_correct = (s_p > s_n)
                if is_correct:
                    correct_rankings += 1
                total_triplets += 1
                
                pos_similarities.append(float(s_p))
                neg_similarities.append(float(s_n))
                margins.append(float(s_p - s_n))
                
                if neg_type in cat_correct:
                    cat_total[neg_type] += 1
                    if is_correct:
                        cat_correct[neg_type] += 1
                        
    # Compute overall classification metrics
    cls_metrics = compute_metrics(all_y_true, all_y_prob)
    
    # Compute margins distribution
    margins = np.array(margins)
    margin_percentiles = {
        "p10": float(np.percentile(margins, 10)),
        "p50": float(np.percentile(margins, 50)),
        "p90": float(np.percentile(margins, 90))
    }
    
    cat_accs = {}
    for c in categories:
        cat_accs[c] = float(cat_correct[c] / cat_total[c]) if cat_total[c] > 0 else 0.0
        
    ranking_accuracy = float(correct_rankings / total_triplets) if total_triplets > 0 else 0.0
    
    metrics = {
        "model_name": model_name,
        "classification": cls_metrics,
        "ranking": {
            "top1_ranking_accuracy": ranking_accuracy,
            "pos_outranks_neg_rate": ranking_accuracy,
            "mean_pos_similarity": float(np.mean(pos_similarities)),
            "mean_neg_similarity": float(np.mean(neg_similarities)),
            "margin_distribution": margin_percentiles,
            "category_accuracy": cat_accs
        }
    }
    return metrics

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V1 vs V2 Siamese Matchers")
    parser.add_argument("--checkpoint", default="models/dl_matcher_v2/best_model_v2.pth", help="Path to V2 checkpoint")
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Evaluation Device: {device}")
    
    dataset_dir = "data/ml_dataset_v2"
    dev_dataset = WaferTripletDataset(
        metadata_path=os.path.join(dataset_dir, "metadata_dev.json"),
        image_dir=dataset_dir
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )
    
    # 1. Load Model V1
    model_v1 = SiameseMatcher(backbone_name="resnet18", pretrained=False)
    v1_ckpt_path = "models/dl_matcher/best_model.pth"
    if os.path.exists(v1_ckpt_path):
        model_v1.load_state_dict(torch.load(v1_ckpt_path, map_location=device))
        print("Model V1 loaded successfully.")
    else:
        print(f"Warning: Model V1 checkpoint not found at {v1_ckpt_path}.")
    model_v1 = model_v1.to(device)
    
    # 2. Load Model V2
    model_v2 = SiameseMatcherV2(backbone_name="resnet18", pretrained=False)
    if os.path.exists(args.checkpoint):
        model_v2.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print("Model V2 loaded successfully.")
    else:
        print(f"Warning: Model V2 checkpoint not found at {args.checkpoint}.")
    model_v2 = model_v2.to(device)
    
    # Run evaluation
    metrics_v1 = run_model_evaluation(model_v1, dev_loader, device, "DL Matcher V1")
    metrics_v2 = run_model_evaluation(model_v2, dev_loader, device, "DL Matcher V2")
    
    # Print Side-By-Side Comparison Table
    print("\n" + "="*80)
    print("DIRECT COMPARISON: DL MATCHER V1 vs V2")
    print("="*80)
    print(f"{'Metric':<35} | {'DL Matcher V1':<15} | {'DL Matcher V2':<15}")
    print("-"*72)
    
    print(f"{'Pair Accuracy':<35} | {metrics_v1['classification']['accuracy']:<15.4f} | {metrics_v2['classification']['accuracy']:<15.4f}")
    print(f"{'F1 Score':<35} | {metrics_v1['classification']['f1']:<15.4f} | {metrics_v2['classification']['f1']:<15.4f}")
    print(f"{'ROC-AUC':<35} | {metrics_v1['classification']['roc_auc']:<15.4f} | {metrics_v2['classification']['roc_auc']:<15.4f}")
    print(f"{'Top-1 Ranking Accuracy':<35} | {metrics_v1['ranking']['top1_ranking_accuracy']:<15.4f} | {metrics_v2['ranking']['top1_ranking_accuracy']:<15.4f}")
    print(f"{'Positive > Hard Negative Rate':<35} | {metrics_v1['ranking']['pos_outranks_neg_rate']:<15.4f} | {metrics_v2['ranking']['pos_outranks_neg_rate']:<15.4f}")
    print(f"{'Random Negative Accuracy':<35} | {metrics_v1['ranking']['category_accuracy']['random']:<15.4f} | {metrics_v2['ranking']['category_accuracy']['random']:<15.4f}")
    print(f"{'Nearby Negative Accuracy':<35} | {metrics_v1['ranking']['category_accuracy']['nearby']:<15.4f} | {metrics_v2['ranking']['category_accuracy']['nearby']:<15.4f}")
    print(f"{'Repeated Negative Accuracy':<35} | {metrics_v1['ranking']['category_accuracy']['repeated']:<15.4f} | {metrics_v2['ranking']['category_accuracy']['repeated']:<15.4f}")
    print(f"{'Wrong Geometry Accuracy':<35} | {metrics_v1['ranking']['category_accuracy']['wrong_geom']:<15.4f} | {metrics_v2['ranking']['category_accuracy']['wrong_geom']:<15.4f}")
    print(f"{'Classical Hard Negative Accuracy':<35} | {metrics_v1['ranking']['category_accuracy']['classical_hard']:<15.4f} | {metrics_v2['ranking']['category_accuracy']['classical_hard']:<15.4f}")
    
    print("\n" + "="*80)
    print("SIMILARITY MARGIN DISTRIBUTION")
    print("="*80)
    print(f"V1: Mean Pos Sim={metrics_v1['ranking']['mean_pos_similarity']:.4f}, Mean Neg Sim={metrics_v1['ranking']['mean_neg_similarity']:.4f}")
    print(f"    Margin Percentiles: P10={metrics_v1['ranking']['margin_distribution']['p10']:.4f}, P50={metrics_v1['ranking']['margin_distribution']['p50']:.4f}, P90={metrics_v1['ranking']['margin_distribution']['p90']:.4f}")
    print(f"V2: Mean Pos Sim={metrics_v2['ranking']['mean_pos_similarity']:.4f}, Mean Neg Sim={metrics_v2['ranking']['mean_neg_similarity']:.4f}")
    print(f"    Margin Percentiles: P10={metrics_v2['ranking']['margin_distribution']['p10']:.4f}, P50={metrics_v2['ranking']['margin_distribution']['p50']:.4f}, P90={metrics_v2['ranking']['margin_distribution']['p90']:.4f}")
    
    # Save metrics JSON files
    model_save_dir = "models/dl_matcher_v2"
    with open(os.path.join(model_save_dir, "development_metrics_v2.json"), "w") as f:
        json.dump(metrics_v2["classification"], f, indent=4)
    with open(os.path.join(model_save_dir, "ranking_metrics_v2.json"), "w") as f:
        json.dump(metrics_v2["ranking"], f, indent=4)
        
if __name__ == "__main__":
    main()
