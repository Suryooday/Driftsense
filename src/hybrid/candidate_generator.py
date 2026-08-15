import numpy as np
import cv2
import time
from typing import List, Dict, Any, Tuple

from src.matching.classical_matcher import (
    _build_rotated_template,
    _extract_top_k_peaks,
    _parabolic_subpixel
)

class CandidateGenerator:
    """
    Generates distinct Top-K candidates for hybrid classical + DL matching.
    
    Performs:
      1. Coarse pose sweep
      2. Spatial NMS to extract distinct peaks across all sweep maps
      3. Local fine sweep around each candidate
      4. Sub-pixel parabolic refinement
    """
    def __init__(
        self,
        zoom_ratio: float = 5.0,
        rot_range_deg: Tuple[float, float] = (-3.0, 3.0),
        scale_range: Tuple[float, float] = (0.97, 1.03),
        rot_steps_coarse: int = 13,
        scale_steps_coarse: int = 7,
        nms_radius: int = 20,
    ) -> None:
        self.zoom_ratio = zoom_ratio
        self.rot_range = rot_range_deg
        self.scale_range = scale_range
        self.rot_steps_coarse = rot_steps_coarse
        self.scale_steps_coarse = scale_steps_coarse
        self.nms_radius = nms_radius

    def generate_candidates(
        self,
        reference: np.ndarray,
        search: np.ndarray,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Generates Top-K refined candidates from the search image.
        """
        if reference.dtype != np.uint8:
            reference = reference.astype(np.uint8)
        if search.dtype != np.uint8:
            search = search.astype(np.uint8)

        sh, sw = search.shape

        # 1. Coarse sweep parameters
        rot_coarse = np.linspace(self.rot_range[0], self.rot_range[1], self.rot_steps_coarse)
        scale_coarse = np.linspace(self.scale_range[0], self.scale_range[1], self.scale_steps_coarse)

        all_coarse_candidates = []

        # We collect coarse peaks
        for angle in rot_coarse:
            for drift_s in scale_coarse:
                s2s = 1.0 / (drift_s * self.zoom_ratio)
                tmpl = _build_rotated_template(reference, angle, s2s)
                th, tw = tmpl.shape
                if tw < 3 or th < 3 or tw >= sw or th >= sh:
                    continue
                
                corr = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
                
                # Extract up to K peaks from this correlation map
                peaks = _extract_top_k_peaks(corr, k=k, min_distance=self.nms_radius)
                
                for row, col, score in peaks:
                    x_center = col + tw / 2.0
                    y_center = row + th / 2.0
                    all_coarse_candidates.append({
                        "x": x_center,
                        "y": y_center,
                        "rotation": angle,
                        "scale": drift_s * self.zoom_ratio,
                        "classical_score": score,
                        "row": row,
                        "col": col,
                        "th": th,
                        "tw": tw,
                        "drift_scale": drift_s
                    })

        if not all_coarse_candidates:
            return []

        # 2. Global spatial NMS across all coarse poses
        # Sort candidates by score descending
        all_coarse_candidates.sort(key=lambda x: x["classical_score"], reverse=True)

        selected_coarse = []
        for cand in all_coarse_candidates:
            if len(selected_coarse) >= k:
                break
            
            # Check spatial separation from already selected ones
            is_distinct = True
            for sel in selected_coarse:
                dist = np.sqrt((cand["x"] - sel["x"]) ** 2 + (cand["y"] - sel["y"]) ** 2)
                if dist < self.nms_radius:
                    is_distinct = False
                    break
            
            if is_distinct:
                selected_coarse.append(cand)

        # 3. Fine sweep & subpixel refinement for each selected candidate locally
        refined_candidates = []
        
        # Calculate steps for fine sweep bounds
        rot_step_c = (self.rot_range[1] - self.rot_range[0]) / max(1, self.rot_steps_coarse - 1)
        scale_step_c = (self.scale_range[1] - self.scale_range[0]) / max(1, self.scale_steps_coarse - 1)

        for rank, cc in enumerate(selected_coarse, 1):
            c_angle = cc["rotation"]
            c_drift_s = cc["drift_scale"]
            c_row = cc["row"]
            c_col = cc["col"]
            c_th = cc["th"]
            c_tw = cc["tw"]

            # Define local search crop with small margin to run template matching fast
            margin = 10
            r0 = max(0, c_row - margin)
            r1 = min(sh, c_row + c_th + margin)
            c0 = max(0, c_col - margin)
            c1 = min(sw, c_col + c_tw + margin)

            crop = search[r0:r1, c0:c1]
            if crop.size == 0 or crop.shape[0] < c_th or crop.shape[1] < c_tw:
                # Fallback to coarse candidate if crop is invalid
                refined_candidates.append({
                    "rank_before_dl": rank,
                    "x": cc["x"],
                    "y": cc["y"],
                    "rotation": cc["rotation"],
                    "scale": cc["scale"],
                    "classical_score": cc["classical_score"]
                })
                continue

            rot_fine = np.linspace(
                max(self.rot_range[0], c_angle - rot_step_c),
                min(self.rot_range[1], c_angle + rot_step_c),
                13
            )
            scale_fine = np.linspace(
                max(self.scale_range[0], c_drift_s - scale_step_c),
                min(self.scale_range[1], c_drift_s + scale_step_c),
                9
            )

            best_score_f = -2.0
            best_angle_f = c_angle
            best_scale_f = c_drift_s
            best_tw_f = c_tw
            best_th_f = c_th
            best_corr_f = None
            best_loc_f = (0, 0)

            for angle in rot_fine:
                for drift_s in scale_fine:
                    s2s = 1.0 / (drift_s * self.zoom_ratio)
                    tmpl = _build_rotated_template(reference, angle, s2s)
                    th, tw = tmpl.shape
                    if tw < 3 or th < 3 or tw >= crop.shape[1] or th >= crop.shape[0]:
                        continue
                    
                    corr = cv2.matchTemplate(crop, tmpl, cv2.TM_CCOEFF_NORMED)
                    _, peak_val, _, peak_loc = cv2.minMaxLoc(corr)
                    if peak_val > best_score_f:
                        best_score_f = peak_val
                        best_corr_f = corr
                        best_angle_f = angle
                        best_scale_f = drift_s
                        best_tw_f = tw
                        best_th_f = th
                        best_loc_f = peak_loc

            if best_corr_f is None:
                # Fallback
                refined_candidates.append({
                    "rank_before_dl": rank,
                    "x": cc["x"],
                    "y": cc["y"],
                    "rotation": cc["rotation"],
                    "scale": cc["scale"],
                    "classical_score": cc["classical_score"]
                })
                continue

            # Subpixel refinement on fine map
            f_col, f_row = best_loc_f
            dx, dy = _parabolic_subpixel(best_corr_f, f_row, f_col)

            # Map back to search coordinate frame
            win_row = r0 + f_row
            win_col = c0 + f_col

            pred_x = (win_col + dx) + best_tw_f / 2.0
            pred_y = (win_row + dy) + best_th_f / 2.0

            refined_candidates.append({
                "rank_before_dl": rank,
                "x": float(pred_x),
                "y": float(pred_y),
                "rotation": float(best_angle_f),
                "scale": float(best_scale_f * self.zoom_ratio),
                "classical_score": float(best_score_f)
            })

        return refined_candidates
