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
        
        # 1. Candidate Generation (Classical)
        candidates = self.candidate_generator.generate_candidates(reference, search, k=self.num_candidates_k)
        
        if not candidates:
            t_total = time.perf_counter() - t_start
            return {
                "predicted_x": None,
                "predicted_y": None,
                "predicted_rotation": None,
                "predicted_scale": None,
                "confidence_score": 0.0,
                "elapsed_s": round(t_total, 4)
            }
            
        # 2. Select candidate using classical scoring only (rank_before_dl == 1)
        best_cand = None
        for c in candidates:
            if c["rank_before_dl"] == 1:
                best_cand = c
                break
        if best_cand is None:
            best_cand = candidates[0]
            
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
        drift_s_center = refined_scale / self.zoom_ratio
        ds_span = self.ref_cfg["scale"]["drift_span"]
        s_steps = self.ref_cfg["scale"]["scale_steps"]
        drift_scale_grid = np.linspace(drift_s_center - ds_span, drift_s_center + ds_span, s_steps)
        for ds in drift_scale_grid:
            s = ds * self.zoom_ratio
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
                
        t_total = time.perf_counter() - t_start
        
        return {
            "predicted_x": float(best_cand["x"]),
            "predicted_y": float(best_cand["y"]),
            "predicted_rotation": float(refined_rot),
            "predicted_scale": float(refined_scale),
            "confidence_score": float(best_val),
            "elapsed_s": float(t_total)
        }
