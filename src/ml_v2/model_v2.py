import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple

class SiameseMatcherV2(nn.Module):
    """
    Siamese network V2 for matching semiconductor wafer patches.
    Optimized for triplet ranking loss and match probability calibration.
    """
    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        embedding_dim: int = 128
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim
        
        # Load vision backbone
        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            in_features = base.fc.in_features
            base.fc = nn.Identity()
            self.encoder = base
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
            
        # Projection head to get a compact normalized embedding
        self.projector = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim)
        )
        
        # Classification head to produce match probability
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        
    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        features = self.encoder(x)
        embeddings = self.projector(features)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def forward(self, ref: torch.Tensor, cand: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes match probability and cosine similarity for a pair of images.
        """
        emb_ref = self.forward_once(ref)
        emb_cand = self.forward_once(cand)
        
        # Cosine similarity
        similarity = torch.sum(emb_ref * emb_cand, dim=1, keepdim=True)
        
        # Binary classification feature combination
        diff = torch.abs(emb_ref - emb_cand)
        mult = emb_ref * emb_cand
        combined = torch.cat([diff, mult], dim=1)
        
        logits = self.classifier(combined)
        probability = torch.sigmoid(logits)
        
        return probability, similarity
