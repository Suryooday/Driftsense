"""
Drift Sense — Siamese Model Evaluation Script.
"""

import os
import json
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

from src.ml.model import SiameseMatcher
from src.ml.dataset import WaferPairDataset

def evaluate_on_loader(
    model: SiameseMatcher,
    loader: DataLoader,
    device: torch.device
):
    model.eval()
    all_probs = []
    all_labels = []
    all_neg_types = []
    
    with torch.no_grad():
        for ref, cand, labels, metas in loader:
            ref, cand = ref.to(device), cand.to(device)
            probs, sims = model(ref, cand)
            
            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())
            
            # Record negative categories
            for m in metas:
                all_neg_types.append(metas.get("neg_type", ["none"])[0])
                
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= 0.5).astype(np.float32)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)
    acc = np.mean(all_preds == all_labels)
    
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        roc_auc = 0.0
        
    cm = confusion_matrix(all_labels, all_preds)
    
    num_pos = int(np.sum(all_labels == 1))
    num_neg = int(np.sum(all_labels == 0))
    pred_pos = int(np.sum(all_preds == 1))
    pred_neg = int(np.sum(all_preds == 0))
    
    results = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "num_pos": num_pos,
        "num_neg": num_neg,
        "pred_pos": pred_pos,
        "pred_neg": pred_neg
    }
    
    # Hard negative analysis
    hard_neg_metrics = {}
    for neg_type in ["random", "nearby", "repeated", "wrong_geom"]:
        # Find indices corresponding to this negative category + all positives
        indices = []
        for i, label in enumerate(all_labels):
            if label == 1:
                indices.append(i)
            elif label == 0 and all_neg_types[i] == neg_type:
                indices.append(i)
                
        if len(indices) == 0:
            continue
            
        sub_labels = all_labels[indices]
        sub_preds = all_preds[indices]
        
        sub_acc = np.mean(sub_preds == sub_labels)
        hard_neg_metrics[neg_type] = {
            "accuracy": float(sub_acc),
            "count": len(sub_labels) - num_pos
        }
        
    results["hard_negatives"] = hard_neg_metrics
    return results

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Wafer Siamese Matcher")
    parser.add_argument("--config", default="ml_config.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", default="models/dl_matcher/best_model.pth", help="Checkpoint file path")
    parser.add_argument("--split", choices=["val", "dev"], default="dev", help="Dataset split to evaluate")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test")
    args = parser.parse_args()
    
    # Load configuration
    import yaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # Detect device: CUDA -> MPS -> CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Device: {device}")
    
    output_dir = config.get("output_dir", "data/ml_dataset")
    meta_path = os.path.join(output_dir, f"metadata_{args.split}.json")
    
    dataset = WaferPairDataset(meta_path, output_dir)
    if args.smoke_test:
        dataset.pairs = dataset.pairs[:32]
        
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Initialize model
    model = SiameseMatcher(backbone_name="resnet18").to(device)
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print(f"⚠ WARNING: Checkpoint not found at {args.checkpoint}. Running evaluation with random weights.")
        
    results = evaluate_on_loader(model, loader, device)
    
    # Print metrics
    print(f"\n==================================================")
    print(f"  EVALUATION METRICS REPORT — {args.split.upper()} SPLIT")
    print(f"==================================================")
    print(f"Accuracy:                 {results['accuracy']:.4f}")
    print(f"Precision:                {results['precision']:.4f}")
    print(f"Recall:                   {results['recall']:.4f}")
    print(f"F1 Score:                 {results['f1']:.4f}")
    print(f"ROC-AUC:                  {results['roc_auc']:.4f}")
    print(f"Confusion Matrix:")
    print(f"  True Neg | False Pos:   {results['confusion_matrix'][0][0]} | {results['confusion_matrix'][0][1]}")
    print(f"  False Neg | True Pos:   {results['confusion_matrix'][1][0]} | {results['confusion_matrix'][1][1]}")
    print(f"Samples count:            Positives: {results['num_pos']} | Negatives: {results['num_neg']}")
    print(f"Predicted count:          Positives: {results['pred_pos']} | Negatives: {results['pred_neg']}")
    
    print(f"\nHard Negative Breakdown:")
    for neg_type, sub in results["hard_negatives"].items():
        print(f"  - {neg_type:<12} (n={sub['count']}): Accuracy = {sub['accuracy']:.4f}")
    print(f"==================================================")
    
    # Save validation/development results
    out_metric_name = f"{args.split}_metrics.json"
    out_metric_path = os.path.join("models/dl_matcher", out_metric_name)
    with open(out_metric_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Metrics saved to {out_metric_path}")

if __name__ == "__main__":
    main()
