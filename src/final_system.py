"""
Frozen final production system entry point.
Implements: Classical NCC-based candidate matching + high-resolution coordinate-descent pose refinement.
Does not import or use PyTorch/DL components.
"""
import os
import json
import time
import numpy as np
import cv2
from typing import Dict, Any

from src.hybrid.candidate_generator import CandidateGenerator
from src.hybrid.patch_extractor import PatchExtractor

def load_final_config(config_path: str = "configs/final_system_config.json") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return json.load(f)

def ncc(im1: np.ndarray, im2: np.ndarray) -> float:
    if im1.shape != im2.shape:
        im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]), interpolation=cv2.INTER_LINEAR)
    im1_f = im1.astype(float) - np.mean(im1)
    im2_f = im2.astype(float) - np.mean(im2)
    denom = np.linalg.norm(im1_f) * np.linalg.norm(im2_f)
    if denom < 1e-8:
        return -1.0
    return float(np.sum(im1_f * im2_f) / denom)

class FinalSystemMatcher:
    def __init__(self, config_path: str = "configs/final_system_config.json") -> None:
        self.config = load_final_config(config_path)
        self.zoom_ratio = self.config["zoom_ratio"]
        
        cg_cfg = self.config["candidate_generation"]
        self.candidate_generator = CandidateGenerator(
            zoom_ratio=self.zoom_ratio,
            rot_range_deg=tuple(cg_cfg["rot_range_deg"]),
            scale_range=tuple(cg_cfg["scale_range"]),
            rot_steps_coarse=cg_cfg["rot_steps_coarse"],
            scale_steps_coarse=cg_cfg["scale_steps_coarse"],
            nms_radius=cg_cfg["nms_radius"]
        )
        self.patch_extractor = PatchExtractor(target_size=self.config["reference_size"][0])
        self.num_candidates_k = cg_cfg["num_candidates_k"]
        self.ref_cfg = self.config["pose_refinement"]
        
    def match(self, reference: np.ndarray, search: np.ndarray, sample_id: str = "unknown") -> Dict[str, Any]:
        t_start = time.perf_counter()
        
        # Adaptive zoom ratio: 10.0 for ~1000x1000 search images, 5.0 for ~512x512 search images
        sh, sw = search.shape[:2]
        if sh >= 800:
            effective_zoom = 10.0
        else:
            effective_zoom = self.zoom_ratio
            
        self.candidate_generator.zoom_ratio = effective_zoom
        # Enforce that the final scale is in [8.0, 12.0]
        self.candidate_generator.scale_range = (8.0 / effective_zoom, 12.0 / effective_zoom)
        
        # Expand crop to cover realistic stage drift (±100 px from center = 200x200 window)
        # This catches targets up to 100 px from the expected stage coordinate.
        crop_h, crop_w = 200, 200
        y0 = max(0, sh // 2 - crop_h // 2)
        x0 = max(0, sw // 2 - crop_w // 2)
        y1 = min(sh, sh // 2 + crop_h // 2)
        x1 = min(sw, sw // 2 + crop_w // 2)
        
        search_crop = search[y0:y1, x0:x1]
        
        # 1. Candidate Generation (Classical) inside the expected drift zone
        candidates = self.candidate_generator.generate_candidates(reference, search_crop, k=self.num_candidates_k)
        
        # Adjust candidate coordinates back to original search space
        for c in candidates:
            c["x"] += x0
            c["y"] += y0
        
        if not candidates:
            t_total = time.perf_counter() - t_start
            return {
                "predicted_x": 0.0,
                "predicted_y": 0.0,
                "predicted_rotation": 0.0,
                "predicted_scale": 0.0,
                "confidence_score": 0.0,
                "found": 0,
                "elapsed_s": round(t_total, 4)
            }
            
        center_x, center_y = sw / 2.0, sh / 2.0
        
        # 2. NEAREST-TO-CENTER selection rule:
        #    When multiple identical patterns exist, always prefer the one closest
        #    to the expected stage coordinate (image center). This is the physically
        #    correct rule: the stage navigator reports the expected position, so the
        #    true target must be the instance nearest to that expected coordinate.
        #
        #    Stage 1: Find the best candidate within the primary drift zone (50 px).
        #             Pick the nearest-to-center among those with score >= 60% of max.
        max_score = max(c["classical_score"] for c in candidates)
        min_score_gate = max_score * 0.60  # loose gate — only filters noise outliers
        
        viable = [c for c in candidates if c["classical_score"] >= min_score_gate]
        
        primary_zone_radius = 60.0  # pixels — must be inside to be considered "primary"
        primary_candidates = [
            c for c in viable
            if np.sqrt((c["x"] - center_x) ** 2 + (c["y"] - center_y) ** 2) <= primary_zone_radius
        ]
        
        if primary_candidates:
            # Multiple instances scenario: pick nearest to center
            best_cand = min(
                primary_candidates,
                key=lambda c: (c["x"] - center_x) ** 2 + (c["y"] - center_y) ** 2
            )
        else:
            # Stage 2: Nothing found within 60px — expand search to full crop region.
            #          Again pick nearest-to-center (not highest score) to enforce the rule.
            extended_zone_radius = 110.0
            extended_candidates = [
                c for c in viable
                if np.sqrt((c["x"] - center_x) ** 2 + (c["y"] - center_y) ** 2) <= extended_zone_radius
            ]
            if extended_candidates:
                best_cand = min(
                    extended_candidates,
                    key=lambda c: (c["x"] - center_x) ** 2 + (c["y"] - center_y) ** 2
                )
            else:
                # Blind search / rejection: absolutely no candidate near center.
                # Pick globally highest NCC score and let the found_threshold decide.
                best_cand = max(candidates, key=lambda c: c["classical_score"])
            
        # 3. High-resolution coordinate descent pose refinement
        refined_rot = best_cand["rotation"]
        refined_scale = best_cand["scale"]
        
        # Iteration 1: Optimize rotation
        best_val = -1.0
        r_span = self.ref_cfg["iteration_1"]["rot_span_deg"]
        r_steps = self.ref_cfg["iteration_1"]["rot_steps"]
        rot_grid = np.linspace(best_cand["rotation"] - r_span, best_cand["rotation"] + r_span, r_steps)
        for r in rot_grid:
            patch = self.patch_extractor.extract_candidate_patch(
                search, best_cand["x"], best_cand["y"], r, refined_scale
            )
            val = ncc(reference, patch)
            if val > best_val:
                best_val = val
                refined_rot = r
                
        # Optimize scale
        drift_s_center = refined_scale / effective_zoom
        ds_span = self.ref_cfg["scale"]["drift_span"]
        s_steps = self.ref_cfg["scale"]["scale_steps"]
        drift_scale_grid = np.linspace(drift_s_center - ds_span, drift_s_center + ds_span, s_steps)
        for ds in drift_scale_grid:
            s = ds * effective_zoom
            patch = self.patch_extractor.extract_candidate_patch(
                search, best_cand["x"], best_cand["y"], refined_rot, s
            )
            val = ncc(reference, patch)
            if val > best_val:
                best_val = val
                refined_scale = s
                
        # Iteration 2: Local fine rotation search around new best
        r2_span = self.ref_cfg["iteration_2"]["rot_span_deg"]
        r2_steps = self.ref_cfg["iteration_2"]["rot_steps"]
        rot_grid_fine = np.linspace(refined_rot - r2_span, refined_rot + r2_span, r2_steps)
        for r in rot_grid_fine:
            patch = self.patch_extractor.extract_candidate_patch(
                search, best_cand["x"], best_cand["y"], r, refined_scale
            )
            val = ncc(reference, patch)
            if val > best_val:
                best_val = val
                refined_rot = r
                
        # ─── 4. Secondary patch verification (SSIM) ──────────────────────────────
        # Extract the best-aligned patch from the FULL search image at refined pose
        best_patch = self.patch_extractor.extract_candidate_patch(
            search, best_cand["x"], best_cand["y"], refined_rot, refined_scale
        )
        
        # Resize to reference size if needed
        ref_h, ref_w = reference.shape[:2]
        if best_patch.shape != reference.shape:
            best_patch = cv2.resize(best_patch, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)
        
        # Compute SSIM manually (no scipy dependency)
        def _ssim(img1: np.ndarray, img2: np.ndarray) -> float:
            i1 = img1.astype(np.float64)
            i2 = img2.astype(np.float64)
            C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
            mu1 = cv2.GaussianBlur(i1, (11, 11), 1.5)
            mu2 = cv2.GaussianBlur(i2, (11, 11), 1.5)
            mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
            s1 = cv2.GaussianBlur(i1 ** 2, (11, 11), 1.5) - mu1_sq
            s2 = cv2.GaussianBlur(i2 ** 2, (11, 11), 1.5) - mu2_sq
            s12 = cv2.GaussianBlur(i1 * i2, (11, 11), 1.5) - mu1_mu2
            num = (2 * mu1_mu2 + C1) * (2 * s12 + C2)
            den = (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
            ssim_map = num / (den + 1e-10)
            return float(np.mean(ssim_map))
        
        ssim_score = _ssim(reference.astype(np.uint8), best_patch.astype(np.uint8))
        
        # Combined confidence: NCC gives coarse structure match,
        # SSIM captures fine-grained local texture uniqueness.
        combined_score = 0.55 * best_val + 0.45 * ssim_score
        
        t_total = time.perf_counter() - t_start
        
        # ─── 5. Found decision — 3-gate system ───────────────────────────────────
        # Gate 1: NCC coarse threshold (filters gross mismatches and Set B FNs)
        ncc_threshold      = self.config.get("found_threshold",      0.28)
        # Gate 2: Combined NCC+SSIM score (filters low-confidence candidates)
        combined_threshold = self.config.get("combined_threshold",   0.57)
        # Gate 3: SSIM floor — Set C false alarms have SSIM < 0.42 for highly
        #         degraded references even when NCC is ~0.75+
        ssim_floor         = self.config.get("ssim_floor",           0.40)
        
        gate1 = best_val      >= ncc_threshold
        gate2 = combined_score >= combined_threshold
        gate3 = ssim_score    >= ssim_floor
        
        found = 1 if (gate1 and gate2 and gate3) else 0
        
        if found == 1:
            return {
                "predicted_x": float(best_cand["x"]),
                "predicted_y": float(best_cand["y"]),
                "predicted_rotation": float(refined_rot),
                "predicted_scale": float(refined_scale),
                "confidence_score": float(combined_score),
                "ncc_score": float(best_val),
                "ssim_score": float(ssim_score),
                "found": 1,
                "elapsed_s": float(t_total)
            }
        else:
            return {
                "predicted_x": 0.0,
                "predicted_y": 0.0,
                "predicted_rotation": 0.0,
                "predicted_scale": 0.0,
                "confidence_score": float(combined_score),
                "ncc_score": float(best_val),
                "ssim_score": float(ssim_score),
                "found": 0,
                "elapsed_s": float(t_total)
            }

