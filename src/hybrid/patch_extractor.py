import cv2
import numpy as np

class PatchExtractor:
    """
    Extracts a 256x256 candidate patch from the search image centered at (x, y)
    with the predicted rotation and scale, and prepares tensors for PyTorch.
    """
    def __init__(self, target_size: int = 256) -> None:
        self.target_size = target_size

    def extract_candidate_patch(
        self,
        search: np.ndarray,
        x: float,
        y: float,
        rotation: float,
        scale: float
    ) -> np.ndarray:
        """
        Extracts a patch from search image mapping destination patch (256, 256)
        to the search coordinate space centered at (x, y) with the target rotation/scale.
        """
        # destination center is (128, 128)
        dest_cx = self.target_size / 2.0
        dest_cy = self.target_size / 2.0

        # Construct M that maps source coords (px, py) -> destination coords (u, v)
        theta = np.radians(rotation)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        M = np.zeros((2, 3), dtype=np.float32)
        M[0, 0] = scale * cos_t
        M[0, 1] = scale * sin_t
        M[0, 2] = dest_cx - scale * (x * cos_t + y * sin_t)

        M[1, 0] = -scale * sin_t
        M[1, 1] = scale * cos_t
        M[1, 2] = dest_cy - scale * (-x * sin_t + y * cos_t)

        # Extract patch with reflection padding if it falls out of search image bounds
        patch = cv2.warpAffine(
            search,
            M,
            (self.target_size, self.target_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        return patch

    def preprocess_patch(self, patch: np.ndarray):
        """
        Normalizes a grayscale patch to [0, 1] range and converts to (1, 1, 256, 256) tensor.
        """
        import torch
        # Normalize to [0, 1]
        norm_patch = patch.astype(np.float32) / 255.0
        # Add channel and batch dimensions
        norm_patch = np.expand_dims(norm_patch, axis=(0, 1))
        # Convert to tensor
        return torch.from_numpy(norm_patch)
