"""
Drift Sense - Classical and Deep Learning Wafer Matching Pipeline.

Implements feature-based classical matching using SIFT and robust template-based
matching under significant zoom and rotation/scale drift.
"""

import os
import cv2
import yaml
import numpy as np
from typing import Tuple, Dict, Any, Optional


class Matcher:
    """
    Wafer matcher utilizing SIFT matching and scale-space template matching
    to align high-zoom reference images within low-zoom search images.
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Initializes the SIFT detector and loads configuration parameters."""
        self.sift = cv2.SIFT_create()

        # FLANN matcher parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Load zoom_ratio and drift ranges from config
        self.zoom_ratio = 10.0
        self.rot_bounds = [-3.0, 3.0]
        self.scale_bounds = [0.97, 1.03]

        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                self.zoom_ratio = config.get("zoom_ratio", 10.0)
                self.rot_bounds = config.get("rotation_bounds", [-3.0, 3.0])
                self.scale_bounds = config.get("scale_bounds", [0.97, 1.03])

    def template_match(
        self,
        reference_img: np.ndarray,
        search_img: np.ndarray
    ) -> Optional[Dict[str, float]]:
        """
        Performs robust scale-space and rotation-space template matching to locate reference.

        Args:
            reference_img: Grayscale uint8 reference image (high-zoom crop).
            search_img: Grayscale uint8 search image (low-zoom wide-field).

        Returns:
            A dictionary containing predicted pose parameters:
            {
                'pred_x': float,
                'pred_y': float,
                'pred_rot': float,
                'pred_scale': float
            }
        """
        h_ref, w_ref = reference_img.shape
        center_ref = (w_ref / 2.0, h_ref / 2.0)

        best_score = -1.0
        best_pose = None

        # Grid search parameters (coarse-to-fine search)
        # Coarse grid (13 steps for rot, 7 steps for scale)
        rot_coarse = np.linspace(self.rot_bounds[0], self.rot_bounds[1], 13)
        scale_coarse = np.linspace(self.scale_bounds[0], self.scale_bounds[1], 7)

        for rot in rot_coarse:
            for scale_drift in scale_coarse:
                # Combined scale factor to map reference to search scale
                total_scale = scale_drift * self.zoom_ratio
                scale_to_search = 1.0 / total_scale

                # Warped template size at search scale
                w_temp = int(round(w_ref * scale_to_search))
                h_temp = int(round(h_ref * scale_to_search))
                if w_temp < 5 or h_temp < 5:
                    continue

                # Build transformation matrix (rotate reference by -rot to align with search)
                M = cv2.getRotationMatrix2D(center_ref, -rot, scale_to_search)
                M[0, 2] += (w_temp / 2.0) - center_ref[0]
                M[1, 2] += (h_temp / 2.0) - center_ref[1]

                template = cv2.warpAffine(
                    reference_img, M, (w_temp, h_temp), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
                )

                # Match template using normalized cross-correlation
                res = cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                if max_val > best_score:
                    best_score = max_val
                    best_pose = (max_loc, w_temp, h_temp, rot, scale_drift)

        if best_pose is None:
            return None

        max_loc, w_temp, h_temp, best_rot_coarse, best_scale_coarse = best_pose

        # Fine grid search around the best coarse parameters (spacing: 0.1° rot, 0.0025 scale)
        rot_fine = np.linspace(best_rot_coarse - 0.5, best_rot_coarse + 0.5, 11)
        scale_fine = np.linspace(best_scale_coarse - 0.01, best_scale_coarse + 0.01, 9)

        for rot in rot_fine:
            # Clamp to search bounds
            if not (self.rot_bounds[0] <= rot <= self.rot_bounds[1]):
                continue
            for scale_drift in scale_fine:
                if not (self.scale_bounds[0] <= scale_drift <= self.scale_bounds[1]):
                    continue

                total_scale = scale_drift * self.zoom_ratio
                scale_to_search = 1.0 / total_scale

                w_temp = int(round(w_ref * scale_to_search))
                h_temp = int(round(h_ref * scale_to_search))
                if w_temp < 5 or h_temp < 5:
                    continue

                M = cv2.getRotationMatrix2D(center_ref, -rot, scale_to_search)
                M[0, 2] += (w_temp / 2.0) - center_ref[0]
                M[1, 2] += (h_temp / 2.0) - center_ref[1]

                template = cv2.warpAffine(
                    reference_img, M, (w_temp, h_temp), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
                )

                res = cv2.matchTemplate(search_img, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                if max_val > best_score:
                    best_score = max_val
                    best_pose = (max_loc, w_temp, h_temp, rot, scale_drift)

        max_loc, w_temp, h_temp, final_rot, final_scale = best_pose
        pred_x = max_loc[0] + w_temp / 2.0
        pred_y = max_loc[1] + h_temp / 2.0

        return {
            "pred_x": float(pred_x),
            "pred_y": float(pred_y),
            "pred_rot": float(final_rot),
            "pred_scale": float(final_scale * self.zoom_ratio)
        }

    def classical_match(
        self,
        reference_img: np.ndarray,
        search_img: np.ndarray
    ) -> Optional[Dict[str, float]]:
        """
        Performs SIFT feature matching and estimates reference pose inside search image.

        Args:
            reference_img: Grayscale uint8 reference image (high-zoom crop).
            search_img: Grayscale uint8 search image (low-zoom wide-field).

        Returns:
            A dictionary containing predicted pose parameters:
            {
                'pred_x': float,
                'pred_y': float,
                'pred_rot': float,
                'pred_scale': float
            }
            Or None if matching fails (e.g. not enough feature matches).
        """
        # 1. Detect SIFT keypoints and descriptors
        kp_ref, des_ref = self.sift.detectAndCompute(reference_img, None)
        kp_search, des_search = self.sift.detectAndCompute(search_img, None)

        if des_ref is None or des_search is None or len(kp_ref) < 4 or len(kp_search) < 4:
            return None

        # 2. Match descriptors using KNN (k=2)
        matches = self.flann.knnMatch(des_ref, des_search, k=2)

        # 3. Apply Lowe's ratio test to filter matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 4:
            return None

        # 4. Extract matched coordinates
        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_search[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 5. Estimate Homography matrix using RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return None

        # 6. Extract center location in search image coordinate space
        h_ref, w_ref = reference_img.shape
        ref_center = np.array([[[w_ref / 2.0, h_ref / 2.0]]], dtype=np.float32)
        pred_center = cv2.perspectiveTransform(ref_center, H)[0][0]
        pred_x, pred_y = float(pred_center[0]), float(pred_center[1])

        # 7. Extract Scale and Rotation from the Homography matrix
        h00 = H[0, 0]
        h10 = H[1, 0]
        s_x = np.sqrt(h00**2 + h10**2)
        pred_scale = 1.0 / s_x if s_x > 0 else 0.0

        angle_H_rad = np.arctan2(h10, h00)
        pred_rot = -np.degrees(angle_H_rad)
        pred_rot = (pred_rot + 180) % 360 - 180

        return {
            "pred_x": pred_x,
            "pred_y": pred_y,
            "pred_rot": pred_rot,
            "pred_scale": pred_scale
        }
