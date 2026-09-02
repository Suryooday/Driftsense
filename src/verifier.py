"""
Drift-Sense Phase 2 — Label Verification Gate Module (Section 5).

Implements:
1. Disk read-back gate: Reads written PNG files from disk to prevent quantization/intermediate mismatches.
2. Dual independent template renderers:
   - Renderer 1: Supersampled area-averaged forward affine resampler.
   - Renderer 2: Direct box-blur filter followed by standard linear forward affine warp.
3. Normalized Cross-Correlation (NCC) template matching against search image.
4. Peak error assertion (error <= 3.0 px) and secondary peak margin calculation (margin >= 0.02, prefer >= 0.12).
"""

import math
from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2

from src.geometry import get_canvas_to_search_matrix


def render_template_v1(
    ref_img: np.ndarray,
    z: float,
    theta_deg: float
) -> np.ndarray:
    """
    Verifier Renderer 1:
    Warp reference image (1000x1000 at 1 nm/px) down to search raster resolution (z nm/px)
    with rotation theta using high-precision area-averaging.
    """
    h_ref, w_ref = ref_img.shape[:2]
    w_t = int(round(w_ref / z))
    h_t = int(round(h_ref / z))
    c_ref = ((w_ref - 1) / 2.0, (h_ref - 1) / 2.0)

    # Use 2x oversampling for sharp anti-aliasing
    inter_scale = 2
    inter_w = w_t * inter_scale
    inter_h = h_t * inter_scale
    c_inter = ((inter_w - 1) / 2.0, (inter_h - 1) / 2.0)
    inter_z = z / float(inter_scale)

    T_ref2inter = get_canvas_to_search_matrix(c_ref, c_inter, inter_z, theta_deg)

    inter_img = cv2.warpAffine(
        ref_img,
        T_ref2inter[:2, :],
        (inter_w, inter_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
    return cv2.resize(inter_img, (w_t, h_t), interpolation=cv2.INTER_AREA)


def render_template_v2(
    ref_img: np.ndarray,
    z: float,
    theta_deg: float
) -> np.ndarray:
    """
    Verifier Renderer 2 (Deliberately Independent):
    Plain box blur filter (kernel size ~ z) followed by direct linear forward affine warp.
    """
    h_ref, w_ref = ref_img.shape[:2]
    ksize = int(round(z))
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(ksize, 3)
    blurred = cv2.boxFilter(ref_img, -1, (ksize, ksize))

    w_t = int(round(w_ref / z))
    h_t = int(round(h_ref / z))
    c_ref = ((w_ref - 1) / 2.0, (h_ref - 1) / 2.0)
    c_t = ((w_t - 1) / 2.0, (h_t - 1) / 2.0)

    T_ref2t = get_canvas_to_search_matrix(c_ref, c_t, z, theta_deg)

    return cv2.warpAffine(
        blurred,
        T_ref2t[:2, :],
        (w_t, h_t),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )


def match_and_calculate_margin(
    search_gray: np.ndarray,
    template: np.ndarray,
    gt_x: float,
    gt_y: float,
    exclusion_radius_px: float = 15.0
) -> Dict[str, Any]:
    """
    Executes TM_CCOEFF_NORMED matching, locates global peak and competing secondary peak outside
    an exclusion radius around the ground-truth location, and computes center error and margin.
    """
    h_t, w_t = template.shape[:2]
    h_s, w_s = search_gray.shape[:2]

    # Run normalized cross-correlation
    corr_map = cv2.matchTemplate(search_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corr_map)

    # Peak center in search coordinates
    peak_tl_x, peak_tl_y = max_loc
    det_x = float(peak_tl_x + (w_t - 1) / 2.0)
    det_y = float(peak_tl_y + (h_t - 1) / 2.0)

    center_err = float(math.hypot(det_x - gt_x, det_y - gt_y))

    # Mask out the exclusion window around the global peak to find the best competing peak
    corr_masked = corr_map.copy()
    h_m, w_m = corr_masked.shape
    yy, xx = np.mgrid[0:h_m, 0:w_m]
    gt_tl_x = gt_x - (w_t - 1) / 2.0
    gt_tl_y = gt_y - (h_t - 1) / 2.0

    dist_from_gt = np.sqrt((xx - gt_tl_x) ** 2 + (yy - gt_tl_y) ** 2)
    corr_masked[dist_from_gt <= exclusion_radius_px] = -1.0

    _, secondary_peak, _, sec_loc = cv2.minMaxLoc(corr_masked)
    margin = float(max_val - secondary_peak)

    return {
        "global_peak": float(max_val),
        "secondary_peak": float(secondary_peak),
        "margin": margin,
        "det_x": det_x,
        "det_y": det_y,
        "center_error_px": center_err,
        "passes_peak": (center_err <= 3.0)
    }


def verify_disk_pair(
    search_path: str,
    ref_path: str,
    gt_present: int,
    gt_x: float,
    gt_y: float,
    gt_theta: float,
    gt_scale: float,
    margin_floor: float = 0.02
) -> Dict[str, Any]:
    """
    Section 5 Verification Gate:
    Re-reads saved PNGs from disk, verifies present pairs with dual independent renderers,
    and returns comprehensive verification diagnostics.
    """
    if gt_present == 0:
        return {
            "verified": True,
            "present": 0,
            "v1_error": 0.0,
            "v1_margin": 1.0,
            "v2_error": 0.0,
            "v2_margin": 1.0,
            "reason": "Absent pair certified"
        }

    search_img = cv2.imread(search_path, cv2.IMREAD_UNCHANGED)
    ref_img = cv2.imread(ref_path, cv2.IMREAD_UNCHANGED)

    if search_img is None or ref_img is None:
        return {
            "verified": False,
            "reason": f"Failed to read disk files: {search_path} or {ref_path}"
        }

    # If 3-channel optical, convert to grayscale for NCC matching
    if len(search_img.shape) == 3:
        search_gray = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    else:
        search_gray = search_img
        ref_gray = ref_img

    z = gt_scale
    th = gt_theta

    # Cross-check Verifier 1 (Area-average resampler)
    t1 = render_template_v1(ref_gray, z, th)
    res1 = match_and_calculate_margin(search_gray, t1, gt_x, gt_y)

    # Cross-check Verifier 2 (Box blur + affine warp)
    t2 = render_template_v2(ref_gray, z, th)
    res2 = match_and_calculate_margin(search_gray, t2, gt_x, gt_y)

    # Binding constraints
    v1_pass = res1["passes_peak"] and (res1["margin"] >= margin_floor)
    v2_pass = res2["passes_peak"] and (res2["margin"] >= margin_floor)
    both_pass = v1_pass and v2_pass

    return {
        "verified": both_pass,
        "present": 1,
        "v1_peak": res1["global_peak"],
        "v1_margin": res1["margin"],
        "v1_error": res1["center_error_px"],
        "v2_peak": res2["global_peak"],
        "v2_margin": res2["margin"],
        "v2_error": res2["center_error_px"],
        "reason": "Passed dual verifier gate" if both_pass else f"Gate failure (v1_err={res1['center_error_px']:.2f}, v1_marg={res1['margin']:.3f}, v2_err={res2['center_error_px']:.2f})"
    }
