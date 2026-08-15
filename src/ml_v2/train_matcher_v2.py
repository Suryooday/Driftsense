import os
import json
import time
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional

from src.ml_v2.model_v2 import SiameseMatcherV2

class WaferTripletDataset(Dataset):
    def __init__(self, metadata_path: str, image_dir: str, transform=None) -> None:
        self.image_dir = image_dir
        self.transform = transform
        
        with open(metadata_path, "r") as f:
            self.triplets = json.load(f)
            
        print(f"Pre-loading {len(self.triplets)} triplets from {metadata_path} into memory...")
        self.imgs_ref = []
        self.imgs_pos = []
        self.imgs_neg = []
        
        for idx, meta in enumerate(self.triplets):
            ref_path = os.path.join(self.image_dir, meta["ref_path"])
            pos_path = os.path.join(self.image_dir, meta["pos_path"])
            neg_path = os.path.join(self.image_dir, meta["neg_path"])
            
            ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            pos_img = cv2.imread(pos_path, cv2.IMREAD_GRAYSCALE)
            neg_img = cv2.imread(neg_path, cv2.IMREAD_GRAYSCALE)
            
            if ref_img is None or pos_img is None or neg_img is None:
                ref_img = np.zeros((256, 256), dtype=np.uint8)
                pos_img = np.zeros((256, 256), dtype=np.uint8)
                neg_img = np.zeros((256, 256), dtype=np.uint8)
                
            self.imgs_ref.append(ref_img)
            self.imgs_pos.append(pos_img)
            self.imgs_neg.append(neg_img)
            
            if (idx + 1) % 1000 == 0:
                print(f"  Loaded {idx + 1} / {len(self.triplets)}...")
            
    def __len__(self) -> int:
        return len(self.triplets)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        meta = self.triplets[idx]
        
        ref_img = self.imgs_ref[idx].astype(np.float32) / 255.0
        pos_img = self.imgs_pos[idx].astype(np.float32) / 255.0
        neg_img = self.imgs_neg[idx].astype(np.float32) / 255.0
        
        ref_tensor = torch.from_numpy(np.expand_dims(ref_img, axis=0))
        pos_tensor = torch.from_numpy(np.expand_dims(pos_img, axis=0))
        neg_tensor = torch.from_numpy(np.expand_dims(neg_img, axis=0))
        
        if self.transform is not None:
            ref_tensor = self.transform(ref_tensor)
            pos_tensor = self.transform(pos_tensor)
            neg_tensor = self.transform(neg_tensor)
            
        return ref_tensor, pos_tensor, neg_tensor, meta

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_validation(model, loader, device, margin=0.2):
    model.eval()
    triplet_losses = []
    cls_losses = []
    total_losses = []
    correct_rankings = 0
    total_samples = 0
    
    bce = nn.BCELoss()
    
    with torch.no_grad():
        for ref, pos, neg, _ in loader:
            ref, pos, neg = ref.to(device), pos.to(device), neg.to(device)
            
            prob_pos, sim_pos = model(ref, pos)
            prob_neg, sim_neg = model(ref, neg)
            
            # Loss calculations
            trip_loss = torch.clamp(sim_neg - sim_pos + margin, min=0.0).mean()
            cls_loss = 0.5 * (bce(prob_pos, torch.ones_like(prob_pos)) + bce(prob_neg, torch.zeros_like(prob_neg)))
            loss = trip_loss + cls_loss
            
            triplet_losses.append(trip_loss.item())
            cls_losses.append(cls_loss.item())
            total_losses.append(loss.item())
            
            # Ranking accuracy: positive similarity should exceed negative similarity
            correct_rankings += torch.sum(sim_pos > sim_neg).item()
            total_samples += ref.size(0)
            
    val_metrics = {
        "triplet_loss": float(np.mean(triplet_losses)),
        "cls_loss": float(np.mean(cls_losses)),
        "total_loss": float(np.mean(total_losses)),
        "ranking_accuracy": float(correct_rankings / total_samples) if total_samples > 0 else 0.0
    }
    return val_metrics

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Wafer Siamese Matcher V2")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--margin", type=float, default=0.2, help="Triplet margin")
    parser.add_argument("--lambda-rank", type=float, default=1.0, help="Weight for ranking loss")
    parser.add_argument("--lambda-cls", type=float, default=1.0, help="Weight for classification loss")
    args = parser.parse_args()
    
    set_seed(1000) # train seed 1000
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Selected Device: {device}")
    
    dataset_dir = "data/ml_dataset_v2"
    model_save_dir = "models/dl_matcher_v2"
    os.makedirs(model_save_dir, exist_ok=True)
    
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1)
    ])
    
    train_dataset = WaferTripletDataset(
        metadata_path=os.path.join(dataset_dir, "metadata_train.json"),
        image_dir=dataset_dir,
        transform=train_transform
    )
    val_dataset = WaferTripletDataset(
        metadata_path=os.path.join(dataset_dir, "metadata_val.json"),
        image_dir=dataset_dir
    )
    
    if args.smoke_test:
        print("--- RUNNING IN SMOKE TEST MODE ---")
        train_dataset.triplets = train_dataset.triplets[:8]
        val_dataset.triplets = val_dataset.triplets[:8]
        epochs = 1
        batch_size = 8
    else:
        epochs = args.epochs
        batch_size = args.batch_size
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Initialize Siamese model
    model = SiameseMatcherV2(backbone_name="resnet18", pretrained=True)
    # Freeze encoder backbone
    for param in model.encoder.parameters():
        param.requires_grad = False
    model = model.to(device)
    
    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    bce = nn.BCELoss()
    
    # Save model configuration metadata
    model_config = {
        "backbone": "resnet18",
        "embedding_dim": 128,
        "pretrained": True,
        "margin": args.margin,
        "lambda_rank": args.lambda_rank,
        "lambda_cls": args.lambda_cls,
        "learning_rate": args.lr
    }
    with open(os.path.join(model_save_dir, "model_config_v2.json"), "w") as f:
        json.dump(model_config, f, indent=4)
        
    history = {
        "train_loss": [],
        "train_triplet_loss": [],
        "train_cls_loss": [],
        "val_loss": [],
        "val_ranking_acc": []
    }
    
    best_val_loss = float("inf")
    patience = 3
    epochs_no_improve = 0
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_total_loss = 0.0
        running_trip_loss = 0.0
        running_cls_loss = 0.0
        
        for ref, pos, neg, _ in train_loader:
            ref, pos, neg = ref.to(device), pos.to(device), neg.to(device)
            
            optimizer.zero_grad()
            prob_pos, sim_pos = model(ref, pos)
            prob_neg, sim_neg = model(ref, neg)
            
            trip_loss = torch.clamp(sim_neg - sim_pos + args.margin, min=0.0).mean()
            cls_loss = 0.5 * (bce(prob_pos, torch.ones_like(prob_pos)) + bce(prob_neg, torch.zeros_like(prob_neg)))
            loss = args.lambda_rank * trip_loss + args.lambda_cls * cls_loss
            
            loss.backward()
            optimizer.step()
            
            running_total_loss += loss.item() * ref.size(0)
            running_trip_loss += trip_loss.item() * ref.size(0)
            running_cls_loss += cls_loss.item() * ref.size(0)
            
        epoch_total_loss = running_total_loss / len(train_dataset)
        epoch_trip_loss = running_trip_loss / len(train_dataset)
        epoch_cls_loss = running_cls_loss / len(train_dataset)
        
        # Evaluate validation set
        val_metrics = evaluate_validation(model, val_loader, device, args.margin)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | "
              f"Train Loss: {epoch_total_loss:.4f} (Trip: {epoch_trip_loss:.4f}, Cls: {epoch_cls_loss:.4f}) | "
              f"Val Loss: {val_metrics['total_loss']:.4f} | "
              f"Val Rank Acc: {val_metrics['ranking_accuracy']:.4f}")
              
        history["train_loss"].append(epoch_total_loss)
        history["train_triplet_loss"].append(epoch_trip_loss)
        history["train_cls_loss"].append(epoch_cls_loss)
        history["val_loss"].append(val_metrics["total_loss"])
        history["val_ranking_acc"].append(val_metrics["ranking_accuracy"])
        
        # Early stopping based on validation total loss
        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            torch.save(model.state_dict(), os.path.join(model_save_dir, "best_model_v2.pth"))
            with open(os.path.join(model_save_dir, "validation_metrics_v2.json"), "w") as f:
                json.dump(val_metrics, f, indent=4)
            epochs_no_improve = 0
            print("  --> Saved new best model checkpoint.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience and not args.smoke_test:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed:.1f} seconds.")
    
    # Save training history
    with open(os.path.join(model_save_dir, "training_history_v2.json"), "w") as f:
        json.dump(history, f, indent=4)

if __name__ == "__main__":
    main()
