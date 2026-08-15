import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Dict, Any, Tuple, Optional

class WaferPairDataset(Dataset):
    """
    PyTorch dataset for reading synthetic wafer reference-candidate patch pairs.
    """
    def __init__(
        self,
        metadata_path: str,
        image_dir: str,
        transform: Optional[transforms.Compose] = None
    ) -> None:
        """
        Args:
            metadata_path: Path to the metadata_{split}.json file.
            image_dir: Path to the directory containing image files.
            transform: Optional torchvision transforms to apply to the reference/candidate.
        """
        self.image_dir = image_dir
        self.transform = transform
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")
            
        with open(metadata_path, "r") as f:
            self.pairs = json.load(f)
            
    def __len__(self) -> int:
        return len(self.pairs)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        meta = self.pairs[idx]
        
        ref_path = os.path.join(self.image_dir, meta["ref_path"])
        cand_path = os.path.join(self.image_dir, meta["cand_path"])
        
        # Load grayscale images
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        cand_img = cv2.imread(cand_path, cv2.IMREAD_GRAYSCALE)
        
        # Gracefully handle corrupt or missing files by returning zero tensors
        if ref_img is None or cand_img is None:
            # Create dummy inputs
            ref_tensor = torch.zeros((1, 256, 256), dtype=torch.float32)
            cand_tensor = torch.zeros((1, 256, 256), dtype=torch.float32)
            label_tensor = torch.tensor([0.0], dtype=torch.float32)
            return ref_tensor, cand_tensor, label_tensor, {"corrupt": True}
            
        # Basic normalization to [0, 1] range and conversion to float32
        ref_img = ref_img.astype(np.float32) / 255.0
        cand_img = cand_img.astype(np.float32) / 255.0
        
        # PyTorch expects shape (C, H, W)
        ref_img = np.expand_dims(ref_img, axis=0)
        cand_img = np.expand_dims(cand_img, axis=0)
        
        ref_tensor = torch.from_numpy(ref_img)
        cand_tensor = torch.from_numpy(cand_img)
        
        # Apply torchvision transforms if present
        if self.transform is not None:
            # Transforms usually expect PIL images or Tensors
            ref_tensor = self.transform(ref_tensor)
            cand_tensor = self.transform(cand_tensor)
            
        label = float(meta["label"])
        label_tensor = torch.tensor([label], dtype=torch.float32)
        
        return ref_tensor, cand_tensor, label_tensor, meta
