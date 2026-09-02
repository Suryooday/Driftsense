import os
import sys
import math
import numpy as np
import cv2

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from generate_dataset import (
    generate_wafer_canvas,
    extract_transformed_patch,
    apply_degradations
)
from src.final_system import FinalSystemMatcher, ncc

class GlobalSearchMatcher(FinalSystemMatcher):
    """Subclass of FinalSystemMatcher that disables the center drift crop to search the full image."""
    def match(self, reference: np.ndarray, search: np.ndarray, sample_id: str = "unknown"):
        # We override match to NOT crop the search image
        t_start = cv2.getTickCount()
        
        sh, sw = search.shape[:2]
        if sh >= 800:
            effective_zoom = 10.0
        else:
            effective_zoom = self.zoom_ratio
            
        self.candidate_generator.zoom_ratio = effective_zoom
        self.candidate_generator.scale_range = (8.0 / effective_zoom, 12.0 / effective_zoom)
        
        # 1. Candidate Generation (Full Search Space)
        candidates = self.candidate_generator.generate_candidates(reference, search, k=self.num_candidates_k)
        
        if not candidates:
            return {
                "predicted_x": 0.0, "predicted_y": 0.0, "predicted_rotation": 0.0, "predicted_scale": 0.0,
                "confidence_score": 0.0, "found": 0, "elapsed_s": 0.0
            }
            
        # Select best candidate globally by NCC score
        best_cand = max(candidates, key=lambda c: c["classical_score"])
        
        # 3. High-resolution coordinate descent pose refinement
        refined_rot = best_cand["rotation"]
        refined_scale = best_cand["scale"]
        
        # Optimize rotation
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
                
        # Fine rotation
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
                
        t_total = (cv2.getTickCount() - t_start) / cv2.getTickFrequency()
        return {
            "predicted_x": float(best_cand["x"]),
            "predicted_y": float(best_cand["y"]),
            "predicted_rotation": float(refined_rot),
            "predicted_scale": float(refined_scale),
            "confidence_score": float(best_val),
            "found": 1,
            "elapsed_s": float(t_total)
        }

def main():
    print("Running Multi-Instance Grid Ambiguity Test...")
    rng = np.random.default_rng(999)
    
    # Matching architectures
    gated_matcher = FinalSystemMatcher()
    global_matcher = GlobalSearchMatcher()
    
    search_w, search_h = 1000, 1000
    ref_w, ref_h = 256, 256
    zoom_ratio = 10.0
    
    # We will run 10 test cases on repeating DRAM layouts
    print("\nEvaluating 10 periodic cases:")
    print("-" * 115)
    print(f"{'Case ID':<8} | {'True Pos':<20} | {'Global Search (No Crop)':<30} | {'Gated Search (Drift Crop)':<30} | {'Result'}")
    print("-" * 115)
    
    success_global = 0
    success_gated = 0
    
    for i in range(10):
        # Generate DRAM grid
        density = 0.25
        style = "DRAM"
        
        canvas_w = int(search_w * zoom_ratio) + 1000
        canvas_h = int(search_h * zoom_ratio) + 1000
        
        canvas = generate_wafer_canvas(canvas_w, canvas_h, density, style, rng)
        
        # Center coordinate in canvas space
        search_cx = canvas_w / 2.0
        search_cy = canvas_h / 2.0
        
        # True drift offset (within typical +- 5.0 pixels search space)
        drift_x = rng.uniform(-4.5, 4.5)
        drift_y = rng.uniform(-4.5, 4.5)
        
        ref_cx = search_cx + (drift_x * zoom_ratio)
        ref_cy = search_cy + (drift_y * zoom_ratio)
        
        # Generate clean images
        search_canvas_w = int(search_w * zoom_ratio)
        search_canvas_h = int(search_h * zoom_ratio)
        
        search_img_clean = extract_transformed_patch(canvas, (search_cx, search_cy), (search_canvas_w, search_canvas_h), 0.0, 1.0)
        search_img_clean = cv2.resize(search_img_clean, (search_w, search_h), interpolation=cv2.INTER_AREA)
        
        sample_scale = rng.uniform(0.9, 1.1)
        sample_rot = rng.uniform(-3.0, 3.0)
        ref_img_clean = extract_transformed_patch(canvas, (ref_cx, ref_cy), (ref_w, ref_h), sample_rot, sample_scale)
        
        # Apply standard noise and charging to simulate realism
        search_img, _ = apply_degradations(search_img_clean, noise_std=0.03, speckle_std=0.01, blur_sigma=1.0, charging_amp=40.0, rng=rng)
        ref_img, _ = apply_degradations(ref_img_clean, noise_std=0.03, speckle_std=0.01, blur_sigma=1.0, charging_amp=20.0, rng=rng)
        
        true_x = (ref_cx - (search_cx - search_canvas_w/2.0)) / zoom_ratio
        true_y = (ref_cy - (search_cy - search_canvas_h/2.0)) / zoom_ratio
        
        # Run Matchers
        res_global = global_matcher.match(ref_img, search_img)
        res_gated = gated_matcher.match(ref_img, search_img)
        
        gx, gy = res_global["predicted_x"], res_global["predicted_y"]
        tx, ty = res_gated["predicted_x"], res_gated["predicted_y"]
        
        err_global = math.sqrt((gx - true_x)**2 + (gy - true_y)**2)
        err_gated = math.sqrt((tx - true_x)**2 + (ty - true_y)**2)
        
        is_g_ok = err_global <= 5.0
        is_t_ok = err_gated <= 5.0
        
        if is_g_ok: success_global += 1
        if is_t_ok: success_gated += 1
        
        status_str = "Both OK" if (is_g_ok and is_t_ok) else ("Gated OK (Global Mismatched)" if is_t_ok else "Both Failed")
        
        print(f"Case {i:02d}  | ({true_x:.1f}, {true_y:.1f}) | ({gx:.1f}, {gy:.1f}) [Err: {err_global:5.1f}px] | ({tx:.1f}, {ty:.1f}) [Err: {err_gated:5.1f}px] | {status_str}")
        
    print("-" * 115)
    print(f"Summary: Gated Search Success Rate: {success_gated/10*100:.1f}% | Global Search Success Rate: {success_global/10*100:.1f}%")
    print("-" * 115)

if __name__ == "__main__":
    main()
