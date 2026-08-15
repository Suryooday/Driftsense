# Google Colab Training Guide — Siamese Wafer Matcher

This document details how to run the supervised Siamese DL matching model training on a CUDA GPU runtime in Google Colab.

---

## 1. Setup and Project Upload

1. Zip your project workspace folder (`Semicon`):
   ```bash
   zip -r Semicon.zip Semicon/ -x "Semicon/venv/*" "Semicon/data/ml_dataset/*"
   ```
   *(Note: Avoid archiving the venv folder or the generated images inside `ml_dataset` to keep the archive small. We will regenerate the dataset or transfer it directly.)*

2. Upload `Semicon.zip` to Colab or clone it directly from your repository:
   ```python
   !git clone <your-repository-url>
   ```

3. Extract the project and navigate to the directory:
   ```python
   !unzip -q Semicon.zip
   %cd Semicon
   ```

---

## 2. Install Dependencies

Install the matching model training dependencies in the Colab notebook cell:
```python
!pip install PyYAML opencv-python numpy matplotlib scikit-learn torch torchvision
```

---

## 3. Verify CUDA Availability

Run this python command to verify that Colab is connected to a GPU backend:
```python
import torch
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
```

---

## 4. Run Training

Launch training directly from the project root using:
```python
!python3 -m src.ml.train_matcher --epochs 10 --batch-size 32 --lr 1e-4
```

---

## 5. Evaluate the Best Checkpoint

After training completes, evaluate the saved checkpoint on the validation split or development split:
```python
# Evaluate on Validation split
!python3 -m src.ml.evaluate_matcher --checkpoint models/dl_matcher/best_model.pth --split val

# Evaluate on Development split (run ONCE after final model selection)
!python3 -m src.ml.evaluate_matcher --checkpoint models/dl_matcher/best_model.pth --split dev
```

---

## 6. Saving and Downloading Checkpoints

The training script automatically writes outputs to `models/dl_matcher/`.

To save or download your results from Colab, compress the directory:
```python
!zip -r dl_matcher_results.zip models/dl_matcher/
```

You can then download `dl_matcher_results.zip` directly from Colab's file explorer.

### Expected Output Files in the zip:
* `best_model.pth`: The trained model weights.
* `model_config.json`: Hyperparameter settings and metadata.
* `training_history.json`: Training and validation loss/accuracy across epochs.
* `val_metrics.json`: Final accuracy, recall, F1, and breakdown metrics for the validation set.
* `dev_metrics.json`: Final metrics evaluated once on the development set.
