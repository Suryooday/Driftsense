import os
import torch
import numpy as np
from typing import List, Dict, Any, Tuple

from src.ml.model import SiameseMatcher

class Reranker:
    """
    Loads the frozen Phase 4B Siamese model and performs inference to verify candidate matches.
    """
    def __init__(self, checkpoint_path: str = "models/dl_matcher/best_model.pth") -> None:
        # Detect device: CUDA -> MPS -> CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        print(f"Reranker initialized using device: {self.device}")
        
        self.model = SiameseMatcher(backbone_name="resnet18", pretrained=False)
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            print(f"Successfully loaded model checkpoint from {checkpoint_path}")
        else:
            raise FileNotFoundError(f"DL model checkpoint not found at: {checkpoint_path}")
            
        self.model.to(self.device)
        self.model.eval()

    def verify_candidates(
        self,
        ref_tensor: torch.Tensor,
        cand_tensors: List[torch.Tensor]
    ) -> List[Tuple[float, float]]:
        """
        Runs batched Siamese inference on a single reference patch and a list of candidate patches.
        
        Args:
            ref_tensor: Tensor of shape (1, 1, 256, 256)
            cand_tensors: List of K tensors, each of shape (1, 1, 256, 256)
            
        Returns:
            List of (match_probability, embedding_similarity) tuples.
        """
        if not cand_tensors:
            return []
            
        K = len(cand_tensors)
        
        # Replicate ref_tensor K times along batch dimension: shape (K, 1, 256, 256)
        ref_batch = ref_tensor.repeat(K, 1, 1, 1).to(self.device)
        
        # Concatenate candidate tensors into shape (K, 1, 256, 256)
        cand_batch = torch.cat(cand_tensors, dim=0).to(self.device)
        
        with torch.no_grad():
            probs, sims = self.model(ref_batch, cand_batch)
            
            # Move to CPU
            probs = probs.cpu().numpy().flatten()
            sims = sims.cpu().numpy().flatten()
            
        return [(float(p), float(s)) for p, s in zip(probs, sims)]
