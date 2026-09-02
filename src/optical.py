"""
Drift-Sense Phase 2 — Optical Microscope Analogue Module (Set D).

Implements:
1. Low-resolution optical defocus softening well below SEM resolution limit.
2. 3-channel RGB generation with channel-specific optical gains.
3. Sub-pixel lateral chromatic shifts per channel.
"""

from typing import Tuple
import numpy as np
import cv2


def simulate_optical_image(
    gray_img: np.ndarray,
    rng: np.random.Generator,
    blur_sigma: float = 3.0,
    chromatic_shift_px: float = 0.5
) -> np.ndarray:
    """
    Transforms a high-resolution SEM/wafer grayscale image into a 3-channel (BGR/RGB)
    optical microscope analogue with chromatic aberration and optical softening.
    """
    h, w = gray_img.shape[:2]
    # Soften image well below SEM resolution
    ksize = int(2 * round(3 * blur_sigma) + 1)
    ksize = max(ksize, 3)
    softened = cv2.GaussianBlur(gray_img, (ksize, ksize), blur_sigma).astype(np.float32)

    # 3 channels (B, G, R) with distinct optical color gains
    # Semiconductor wafers under optical inspection typically have golden/oxide and silicon blue tones
    gain_b = float(rng.uniform(0.75, 0.95))
    gain_g = float(rng.uniform(0.90, 1.10))
    gain_r = float(rng.uniform(1.05, 1.30))

    # Sub-pixel lateral shifts for chromatic aberration
    shift_r_x = float(rng.uniform(-chromatic_shift_px, chromatic_shift_px))
    shift_r_y = float(rng.uniform(-chromatic_shift_px, chromatic_shift_px))
    shift_b_x = -shift_r_x * float(rng.uniform(0.8, 1.2))
    shift_b_y = -shift_r_y * float(rng.uniform(0.8, 1.2))

    M_r = np.array([[1.0, 0.0, shift_r_x], [0.0, 1.0, shift_r_y]], dtype=np.float32)
    M_b = np.array([[1.0, 0.0, shift_b_x], [0.0, 1.0, shift_b_y]], dtype=np.float32)

    ch_r = cv2.warpAffine(softened * gain_r, M_r, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    ch_g = softened * gain_g
    ch_b = cv2.warpAffine(softened * gain_b, M_b, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # Base tinting for silicon substrate + metal lines
    out_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    out_bgr[..., 0] = np.clip(ch_b + rng.uniform(10, 30), 0, 255).astype(np.uint8)  # Blue
    out_bgr[..., 1] = np.clip(ch_g + rng.uniform(15, 35), 0, 255).astype(np.uint8)  # Green
    out_bgr[..., 2] = np.clip(ch_r + rng.uniform(25, 50), 0, 255).astype(np.uint8)  # Red

    return out_bgr
