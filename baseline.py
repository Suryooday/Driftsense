"""
Drift-Sense Phase 2 — Naive Baseline Matcher Module (baseline.py).

Implements the brute-force Normalised Cross-Correlation (NCC) baseline matcher over
a coarse grid of scale z (0.5 steps in [8.0, 12.0]) and rotation theta (1.0 degree steps in [-5.0, 5.0]).
"""

import math
import time
from typing import Dict, Any, Tuple
import numpy as np
import cv2

from src.geometry import get_canvas_to_search_matrix


class NaiveBaselineMatcher:
    def __init__(
        self,
        z_min: float = 8.0,
        z_max: float = 12.0,
        z_step: float = 0.5,
        theta_min: float = -5.0,
        theta_max: float = 5.0,
        theta_step: float = 1.0,
        presence_threshold: float = 0.55
    ):
        self.z_grid = np.arange(z_min, z_max + 1e-4, z_step)
        self.theta_grid = np.arange(theta_min, theta_max + 1e-4, theta_step)
        self.presence_threshold = presence_threshold

    def _render_template(self, ref_gray: np.ndarray, z: float, theta_deg: float) -> np.ndarray:
        h_ref, w_ref = ref_gray.shape[:2]
        w_t = max(10, int(round(w_ref / z)))
        h_t = max(10, int(round(h_ref / z)))
        c_ref = ((w_ref - 1) / 2.0, (h_ref - 1) / 2.0)
        c_t = ((w_t - 1) / 2.0, (h_t - 1) / 2.0)

        # Forward affine transform from ref_gray to template
        T_ref2t = get_canvas_to_search_matrix(c_ref, c_t, z, theta_deg)
        return cv2.warpAffine(
            ref_gray,
            T_ref2t[:2, :],
            (w_t, h_t),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

    def match(self, ref_img: np.ndarray, search_img: np.ndarray) -> Dict[str, Any]:
        """
        Executes coarse grid search over (z, theta) and returns best hypothesis.
        """
        if len(search_img.shape) == 3:
            search_gray = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
            ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        else:
            search_gray = search_img
            ref_gray = ref_img

        best_score = -1.0
        best_x = 0.0
        best_y = 0.0
        best_z = 10.0
        best_theta = 0.0

        for z in self.z_grid:
            for theta in self.theta_grid:
                template = self._render_template(ref_gray, z, theta)
                h_t, w_t = template.shape[:2]

                res = cv2.matchTemplate(search_gray, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)

                if max_val > best_score:
                    best_score = float(max_val)
                    best_x = float(max_loc[0] + (w_t - 1) / 2.0)
                    best_y = float(max_loc[1] + (h_t - 1) / 2.0)
                    best_z = float(z)
                    best_theta = float(theta)

        predicted_present = 1 if (best_score >= self.presence_threshold) else 0

        return {
            "predicted_present": predicted_present,
            "predicted_x": best_x if predicted_present else 0.0,
            "predicted_y": best_y if predicted_present else 0.0,
            "predicted_scale": best_z if predicted_present else 0.0,
            "predicted_theta": best_theta if predicted_present else 0.0,
            "confidence_score": best_score
        }


def match_pair(search_path: str, ref_path: str) -> Dict[str, Any]:
    search_img = cv2.imread(search_path, cv2.IMREAD_UNCHANGED)
    ref_img = cv2.imread(ref_path, cv2.IMREAD_UNCHANGED)
    matcher = NaiveBaselineMatcher()
    return matcher.match(ref_img, search_img)
