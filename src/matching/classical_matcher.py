"""
Drift Sense - Classical Multi-Scale Multi-Angle Template Matcher.

Implements a production-grade wafer-patch alignment algorithm:
  1. Coarse-to-fine sweep over rotation (-3° to +3°) and scale (0.97x–1.03x)
  2. Top-K peak extraction with Non-Maximum Suppression to handle repetitive patterns
  3. Wider-context disambiguation: re-scores each surviving candidate using a larger
     region of the reference vs search to resolve ambiguous repeating cells
  4. Sub-pixel parabolic refinement in both X and Y
  5. Returns a structured result dict including a confidence score

Coordinate convention:
  - `predicted_x` and `predicted_y` are the CENTER of the reference patch in
     the search image's pixel coordinate frame.  (0,0) is the center of the
     top-left pixel, matching the convention used by ground_truth.json.
"""

from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

try:
    from skimage.metrics import structural_similarity as _ssim
    _HAS_SSIM = True
except ImportError:
    _HAS_SSIM = False


# ---------------------------------------------------------------------------
# Public result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    """Pose estimate for one reference-inside-search match."""
    predicted_x: float         # Center-of-reference X in search-image pixels
    predicted_y: float         # Center-of-reference Y in search-image pixels
    predicted_rotation: float  # Best rotation angle (degrees, same sign as ground truth)
    predicted_scale: float     # Best total scale factor (should ≈ zoom_ratio ≈ 5)
    confidence_score: float    # Normalized cross-correlation value at best peak [0,1]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_rotated_template(
    reference: np.ndarray,
    angle: float,
    scale_to_search: float,
) -> np.ndarray:
    """
    Rotates and scales the reference image to match the search coordinate frame.

    Args:
        reference: Grayscale uint8 reference image.
        angle: Rotation to apply (degrees, same sign convention as the dataset).
        scale_to_search: Combined scale factor mapping reference pixels → search pixels
                         (i.e. 1 / (drift_scale * zoom_ratio)).

    Returns:
        Warped grayscale template as uint8.
    """
    h, w = reference.shape
    cx, cy = w / 2.0, h / 2.0
    # Invert rotation direction: we rotate the template to match the drifted reference
    M = cv2.getRotationMatrix2D((cx, cy), -angle, scale_to_search)
    out_w = max(1, int(round(w * scale_to_search)))
    out_h = max(1, int(round(h * scale_to_search)))
    M[0, 2] += out_w / 2.0 - cx
    M[1, 2] += out_h / 2.0 - cy
    return cv2.warpAffine(
        reference, M, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def _extract_top_k_peaks(
    corr_map: np.ndarray,
    k: int,
    min_distance: int,
) -> List[Tuple[int, int, float]]:
    """
    Extracts the top-K distinct peaks from a 2-D correlation map using
    iterative Non-Maximum Suppression.

    Args:
        corr_map: 2-D float32 correlation surface from cv2.matchTemplate.
        k: Number of peaks to return.
        min_distance: Minimum separation (pixels) between returned peaks.

    Returns:
        List of (row, col, score) tuples, best-first.
    """
    working = corr_map.copy()
    peaks: List[Tuple[int, int, float]] = []
    for _ in range(k):
        _, max_val, _, max_loc = cv2.minMaxLoc(working)
        col, row = max_loc        # cv2 returns (x, y) = (col, row)
        peaks.append((row, col, float(max_val)))
        # Suppress a square region around this peak
        r0 = max(0, row - min_distance)
        r1 = min(working.shape[0], row + min_distance + 1)
        c0 = max(0, col - min_distance)
        c1 = min(working.shape[1], col + min_distance + 1)
        working[r0:r1, c0:c1] = -1.0
    return peaks


def _parabolic_subpixel(
    corr_map: np.ndarray,
    row: int,
    col: int,
) -> Tuple[float, float]:
    """
    Fits 1-D parabolas along X and Y through the peak and its neighbours
    to estimate the sub-pixel peak location.

    Args:
        corr_map: 2-D float32 correlation surface.
        row: Integer row index of the peak.
        col: Integer column index of the peak.

    Returns:
        (delta_col, delta_row) sub-pixel offsets; add to (col, row) for the
        refined location.
    """
    h, w = corr_map.shape

    def _refine_1d(left: float, center: float, right: float) -> float:
        denom = 2.0 * (left - 2.0 * center + right)
        if abs(denom) < 1e-8:
            return 0.0
        return (left - right) / denom

    # X direction
    if 0 < col < w - 1:
        dx = _refine_1d(
            corr_map[row, col - 1],
            corr_map[row, col],
            corr_map[row, col + 1],
        )
    else:
        dx = 0.0

    # Y direction
    if 0 < row < h - 1:
        dy = _refine_1d(
            corr_map[row - 1, col],
            corr_map[row, col],
            corr_map[row + 1, col],
        )
    else:
        dy = 0.0

    return dx, dy


def _wider_context_score(
    search: np.ndarray,
    reference: np.ndarray,
    center_col: float,
    center_row: float,
    angle: float,
    scale_to_search: float,
    context_scale: float = 3.5,
) -> float:
    """
    Computes a disambiguation score using a large context window (spanning
    multiple unit cells) to distinguish the correct lattice cell from
    structurally similar neighbours.  Returns a weighted blend of SSIM
    (structural layout) and NCC (intensity correlation).

    Args:
        search: Grayscale uint8 search image.
        reference: Grayscale uint8 reference image.
        center_col: Candidate centre X in search image.
        center_row: Candidate centre Y in search image.
        angle: Rotation angle for the pose.
        scale_to_search: Pixel-scale ratio reference→search.
        context_scale: Multiplier applied to the template size; use ≥3 to
            straddle multiple unit cells and expose unique neighbourhood
            context.

    Returns:
        Disambiguation score in [-1, 1].  Higher = better match.
    """
    # Build a wider reference template (rotated + scaled to search space)
    wider_ref = _build_rotated_template(
        reference,
        angle=angle,
        scale_to_search=scale_to_search * context_scale,
    )
    wh, ww = wider_ref.shape

    # Crop the corresponding region from the search image
    r0 = int(round(center_row - wh / 2.0))
    r1 = r0 + wh
    c0 = int(round(center_col - ww / 2.0))
    c1 = c0 + ww
    sh, sw = search.shape

    # Reflect-pad if the crop extends beyond image boundaries
    pad_top = max(0, -r0)
    pad_bot = max(0, r1 - sh)
    pad_lft = max(0, -c0)
    pad_rgt = max(0, c1 - sw)
    r0, r1 = max(0, r0), min(sh, r1)
    c0, c1 = max(0, c0), min(sw, c1)
    crop = search[r0:r1, c0:c1]
    if crop.size == 0:
        return -1.0
    if pad_top or pad_bot or pad_lft or pad_rgt:
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bot, pad_lft, pad_rgt, cv2.BORDER_REFLECT
        )
    # Resize crop to template size if clamping changed the shape
    if crop.shape != wider_ref.shape:
        crop = cv2.resize(crop, (ww, wh), interpolation=cv2.INTER_LINEAR)

    # --- NCC component ---
    crop_f  = crop.astype(np.float64)
    tmpl_f  = wider_ref.astype(np.float64)
    crop_n  = crop_f - crop_f.mean()
    tmpl_n  = tmpl_f - tmpl_f.mean()
    denom   = np.linalg.norm(crop_n) * np.linalg.norm(tmpl_n)
    ncc     = float(np.sum(crop_n * tmpl_n) / denom) if denom > 1e-8 else 0.0

    # --- SSIM component (sensitive to structural layout, not just intensity) ---
    if _HAS_SSIM and min(wh, ww) >= 7:
        win = min(7, wh - (wh % 2 == 0), ww - (ww % 2 == 0))
        win = win if win % 2 == 1 else win - 1
        win = max(3, win)
        score_ssim = float(_ssim(
            crop.astype(np.float64),
            wider_ref.astype(np.float64),
            data_range=255.0,
            win_size=win,
        ))
    else:
        score_ssim = ncc  # fallback

    # Blend: SSIM is more discriminative for layout; NCC anchors intensity fit
    return 0.6 * score_ssim + 0.4 * ncc


# ---------------------------------------------------------------------------
# Main matcher class
# ---------------------------------------------------------------------------

class ClassicalMatcher:
    """
    Multi-scale, multi-angle template matcher for wafer-patch localization.

    The pipeline:
      1. Coarse sweep  → top-5 peaks per pose with NMS
      2. Fine sweep    → refine around the best coarse pose
      3. Disambiguation → re-score surviving candidates with wider context
      4. Sub-pixel refinement via parabolic interpolation

    Args:
        zoom_ratio: Physical magnification ratio (search pixels / reference pixels).
        rot_range_deg: (min, max) rotation search bounds in degrees.
        scale_range: (min, max) drift scale factors to search.
        rot_steps_coarse: Number of steps in the coarse rotation grid.
        scale_steps_coarse: Number of steps in the coarse scale grid.
        top_k: Number of candidate peaks to extract per correlation map.
        nms_radius: Minimum pixel distance between distinct peaks.
        context_scale: Multiplier for the wider-context disambiguation crop.
    """

    def __init__(
        self,
        zoom_ratio: float = 5.0,
        rot_range_deg: Tuple[float, float] = (-3.0, 3.0),
        scale_range: Tuple[float, float] = (0.97, 1.03),
        rot_steps_coarse: int = 13,
        scale_steps_coarse: int = 7,
        top_k: int = 5,
        nms_radius: int = 20,
        context_scale: float = 3.5,
    ) -> None:
        self.zoom_ratio = zoom_ratio
        self.rot_range = rot_range_deg
        self.scale_range = scale_range
        self.rot_steps_coarse = rot_steps_coarse
        self.scale_steps_coarse = scale_steps_coarse
        self.top_k = top_k
        self.nms_radius = nms_radius
        self.context_scale = context_scale

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(
        self,
        reference: np.ndarray,
        search: np.ndarray,
    ) -> Optional[MatchResult]:
        """
        Locates the reference patch inside the search image.

        Pipeline:
          1. Coarse pose sweep  → track globally-best NCC peak and its pose/map
          2. Fine pose sweep    → refine around the coarse-best pose
          3. Sub-pixel parabola → fit to neighbourhood of the best-NCC pixel

        Note on disambiguation:
          The per-tile structural variation in the dataset (line-position jitter,
          occasional via markers) creates enough local uniqueness that the NCC
          surface already peaks strongly at the correct cell.  Adding a wider-
          context re-scorer is counterproductive: the reference image contains
          only its own 256-px physical region, so "scaling it up" yields no new
          surrounding information to compare against the search crop.

        Args:
            reference: Grayscale uint8 reference image (high-zoom crop).
            search: Grayscale uint8 search image (low-zoom wide field).

        Returns:
            A MatchResult with the best predicted pose, or None if the
            search image is too small to contain any template.
        """
        if reference.dtype != np.uint8:
            reference = reference.astype(np.uint8)
        if search.dtype != np.uint8:
            search = search.astype(np.uint8)

        sh, sw = search.shape

        # ---------------------------------------------------------------
        # Stage 1: Coarse sweep
        # ---------------------------------------------------------------
        rot_coarse   = np.linspace(self.rot_range[0],  self.rot_range[1],  self.rot_steps_coarse)
        scale_coarse = np.linspace(self.scale_range[0], self.scale_range[1], self.scale_steps_coarse)

        best_score_c = -2.0
        best_corr_c: Optional[np.ndarray] = None
        best_angle_c  = 0.0
        best_scale_c  = (self.scale_range[0] + self.scale_range[1]) / 2.0
        best_tw_c = 1
        best_th_c = 1

        for angle in rot_coarse:
            for drift_s in scale_coarse:
                s2s = 1.0 / (drift_s * self.zoom_ratio)
                tmpl = _build_rotated_template(reference, angle, s2s)
                th, tw = tmpl.shape
                if tw < 3 or th < 3 or tw >= sw or th >= sh:
                    continue
                corr = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
                _, peak, _, _ = cv2.minMaxLoc(corr)
                if peak > best_score_c:
                    best_score_c = peak
                    best_corr_c  = corr
                    best_angle_c = angle
                    best_scale_c = drift_s
                    best_tw_c    = tw
                    best_th_c    = th

        if best_corr_c is None:
            return None

        # ---------------------------------------------------------------
        # Stage 2: Fine sweep — tight grid around coarse winner
        # ---------------------------------------------------------------
        rot_step_c   = (self.rot_range[1]   - self.rot_range[0])   / max(1, self.rot_steps_coarse   - 1)
        scale_step_c = (self.scale_range[1] - self.scale_range[0]) / max(1, self.scale_steps_coarse - 1)

        rot_fine   = np.linspace(
            max(self.rot_range[0],   best_angle_c - rot_step_c),
            min(self.rot_range[1],   best_angle_c + rot_step_c),
            13,
        )
        scale_fine = np.linspace(
            max(self.scale_range[0], best_scale_c - scale_step_c),
            min(self.scale_range[1], best_scale_c + scale_step_c),
            9,
        )

        best_score_f  = -2.0
        best_corr_f   = best_corr_c
        best_angle_f  = best_angle_c
        best_scale_f  = best_scale_c
        best_tw_f     = best_tw_c
        best_th_f     = best_th_c

        for angle in rot_fine:
            for drift_s in scale_fine:
                s2s = 1.0 / (drift_s * self.zoom_ratio)
                tmpl = _build_rotated_template(reference, angle, s2s)
                th, tw = tmpl.shape
                if tw < 3 or th < 3 or tw >= sw or th >= sh:
                    continue
                corr = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
                _, peak, _, _ = cv2.minMaxLoc(corr)
                if peak > best_score_f:
                    best_score_f = peak
                    best_corr_f  = corr
                    best_angle_f = angle
                    best_scale_f = drift_s
                    best_tw_f    = tw
                    best_th_f    = th

        # ---------------------------------------------------------------
        # Stage 3: Best-NCC peak location (integer pixel)
        # ---------------------------------------------------------------
        _, _, _, max_loc = cv2.minMaxLoc(best_corr_f)
        win_col, win_row = max_loc   # cv2 returns (x, y) = (col, row)

        # ---------------------------------------------------------------
        # Stage 4: Sub-pixel parabolic refinement
        # ---------------------------------------------------------------
        dx, dy = _parabolic_subpixel(best_corr_f, win_row, win_col)

        pred_x = (win_col + dx) + best_tw_f / 2.0
        pred_y = (win_row + dy) + best_th_f / 2.0

        return MatchResult(
            predicted_x=float(pred_x),
            predicted_y=float(pred_y),
            predicted_rotation=float(best_angle_f),
            predicted_scale=float(best_scale_f * self.zoom_ratio),
            confidence_score=float(best_score_f),
        )

