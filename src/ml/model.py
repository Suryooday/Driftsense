import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple

class SiameseMatcher(nn.Module):
    """
    Siamese network for matching semiconductor wafer patches.
    Uses a shared vision backbone to extract feature embeddings from
    both reference and candidate patches, then computes similarity.
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
            # Remove base classification layer
            base.fc = nn.Identity()
            self.encoder = base
        elif backbone_name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            base = models.mobilenet_v3_small(weights=weights)
            in_features = base.classifier[0].in_features
            base.classifier = nn.Identity()
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
        # The input is the absolute difference between the embeddings and their product
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        
    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        # Grayscale images have 1 channel, but pretrained model expects 3 channels
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        features = self.encoder(x)
        embeddings = self.projector(features)
        # Normalize embeddings to unit sphere
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def forward(self, ref: torch.Tensor, cand: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ref: Reference patch tensor of shape (batch, 1, 256, 256)
            cand: Candidate patch tensor of shape (batch, 1, 256, 256)
            
        Returns:
            Tuple containing:
            - Match probability in [0, 1] (batch, 1)
            - Cosine similarity between embeddings (batch, 1)
        """
        emb_ref = self.forward_once(ref)
        emb_cand = self.forward_once(cand)
        
        # Cosine similarity (since embeddings are L2 normalized, it's just dot product)
        similarity = torch.sum(emb_ref * emb_cand, dim=1, keepdim=True)
        
        # Build features for binary classification: concat absolute difference and elementwise product
        diff = torch.abs(emb_ref - emb_cand)
        mult = emb_ref * emb_cand
        combined = torch.cat([diff, mult], dim=1)
        
        logits = self.classifier(combined)
        probability = torch.sigmoid(logits)
        
        return probability, similarity
