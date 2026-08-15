"""
Drift Sense — Siamese Model Supervised Training Loop.
"""

import os
import json
import time
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from src.ml.model import SiameseMatcher
from src.ml.dataset import WaferPairDataset

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Wafer Siamese Matcher")
    parser.add_argument("--config", default="ml_config.yaml", help="Path to config YAML")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of data loader workers")
    args = parser.parse_args()
    
    # Load configuration
    import yaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    seed = config.get("train_seed", 1000)
    set_seed(seed)
    
    # Detect device: CUDA -> MPS -> CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Selected Device: {device}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    print(f"MPS Being Used: {device.type == 'mps'}")
    
    if device.type == "cuda":
        print(f"  GPU name: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
    elif device.type == "mps":
        print("  Using Apple Silicon MPS acceleration.")
        
    output_dir = config.get("output_dir", "data/ml_dataset")
    model_save_dir = "models/dl_matcher"
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Create simple transforms / data augmentation for training
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1)
    ])
    
    # Instantiate datasets
    train_meta_path = os.path.join(output_dir, "metadata_train.json")
    val_meta_path = os.path.join(output_dir, "metadata_val.json")
    
    train_dataset = WaferPairDataset(train_meta_path, output_dir, transform=train_transform)
    val_dataset = WaferPairDataset(val_meta_path, output_dir)
    
    # Limit dataset sizes if running in smoke test mode
    if args.smoke_test:
        print("--- RUNNING IN SMOKE TEST MODE ---")
        train_dataset.pairs = train_dataset.pairs[:8]
        val_dataset.pairs = val_dataset.pairs[:8]
        epochs = 1
        batch_size = 8
        num_workers = 0
    else:
        epochs = args.epochs
        batch_size = args.batch_size
        num_workers = args.num_workers
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )
    
    # Initialize Siamese model
    model = SiameseMatcher(backbone_name="resnet18", pretrained=True).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Keep track of history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": []
    }
    
    best_val_loss = float("inf")
    patience = 3
    epochs_no_improve = 0
    
    # Start training loop
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_i, (ref, cand, labels, _) in enumerate(train_loader):
            if args.smoke_test:
                print(f"Batch data device before transfer: ref={ref.device}, cand={cand.device}, labels={labels.device}")
            ref, cand, labels = ref.to(device), cand.to(device), labels.to(device)
            if args.smoke_test:
                print(f"Batch data device after transfer: ref={ref.device}, cand={cand.device}, labels={labels.device}")
                if torch.backends.mps.is_available() and device.type == "mps":
                    assert ref.device.type == "mps", f"Expected MPS device, got {ref.device.type}"
                    print("--> Confirmed: MPS is being used for training tensors.")
            
            optimizer.zero_grad()
            probs, sims = model(ref, cand)
            if args.smoke_test:
                print(f"Forward pass completed. Outputs on device: probs={probs.device}, sims={sims.device}")
                
            loss = criterion(probs, labels)
            if args.smoke_test:
                print(f"Loss: {loss.item():.4f}")
                assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
                print("--> Confirmed: Loss is finite.")
                
            loss.backward()
            if args.smoke_test:
                print("Backward pass completed.")
                
            optimizer.step()
            if args.smoke_test:
                print("Optimizer update completed.")
                with torch.no_grad():
                    probs_after, _ = model(ref, cand)
                    loss_after = criterion(probs_after, labels)
                    print(f"Loss before update: {loss.item():.4f} | Loss after update: {loss_after.item():.4f}")
            
            running_loss += loss.item() * ref.size(0)
            
            if args.smoke_test:
                print(f"Batch {batch_i}: Loss = {loss.item():.4f}")
                
        epoch_train_loss = running_loss / len(train_loader.dataset)
        
        # Validation evaluation
        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for ref, cand, labels, _ in val_loader:
                ref, cand, labels = ref.to(device), cand.to(device), labels.to(device)
                probs, sims = model(ref, cand)
                loss = criterion(probs, labels)
                val_running_loss += loss.item() * ref.size(0)
                
                preds = (probs >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total
        
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
        
        # Save checkpoints (ensure checkpoints saved on MPS can later be loaded on CPU, MPS, or CUDA)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            # Save best checkpoint
            checkpoint_path = os.path.join(model_save_dir, "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> Saved new best model checkpoint to {checkpoint_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience and not args.smoke_test:
                print("Early stopping triggered due to lack of validation loss improvement.")
                break
                
    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed:.1f} seconds.")
    
    # Save configurations and histories
    model_config = {
        "backbone_name": model.backbone_name,
        "embedding_dim": model.embedding_dim,
        "train_seed": seed,
        "epochs_trained": epoch + 1,
        "learning_rate": args.lr,
        "batch_size": batch_size,
        "device": str(device)
    }
    
    with open(os.path.join(model_save_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)
        
    with open(os.path.join(model_save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)
        
    print("Training configs and histories written successfully.")

if __name__ == "__main__":
    main()
