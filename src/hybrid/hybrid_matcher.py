import time
import numpy as np
from typing import Dict, Any, List, Tuple

from src.hybrid.candidate_generator import CandidateGenerator
from src.hybrid.patch_extractor import PatchExtractor
from src.hybrid.reranker import Reranker

class HybridMatcher:
    """
    Ties together the classical candidate generator, the DL patch extractor/verifier,
    and the score fusion reranker to predict final alignment poses.
    """
    def __init__(
        self,
        zoom_ratio: float = 5.0,
        rot_range_deg: Tuple[float, float] = (-3.0, 3.0),
        scale_range: Tuple[float, float] = (0.97, 1.03),
        rot_steps_coarse: int = 13,
        scale_steps_coarse: int = 7,
        nms_radius: int = 20,
        checkpoint_path: str = "models/dl_matcher/best_model.pth",
    ) -> None:
        self.zoom_ratio = zoom_ratio
        self.candidate_generator = CandidateGenerator(
            zoom_ratio=zoom_ratio,
            rot_range_deg=rot_range_deg,
            scale_range=scale_range,
            rot_steps_coarse=rot_steps_coarse,
            scale_steps_coarse=scale_steps_coarse,
            nms_radius=nms_radius
        )
        self.patch_extractor = PatchExtractor()
        self.reranker = Reranker(checkpoint_path=checkpoint_path)

    def match_hybrid(
        self,
        reference: np.ndarray,
        search: np.ndarray,
        sample_id: str,
        k: int = 5,
        ranking_mode: str = "hybrid",  # "classical", "dl", "hybrid"
        alpha: float = 0.5,
        beta: float = 0.5
    ) -> Dict[str, Any]:
        """
        Executes the hybrid matching pipeline on a single image pair.
        """
        t_start = time.perf_counter()

        # 1. Candidate Generation (Classical)
        t_class_start = time.perf_counter()
        candidates = self.candidate_generator.generate_candidates(reference, search, k=k)
        t_class_end = time.perf_counter()
        elapsed_classical = t_class_end - t_class_start

        if not candidates:
            # Empty fallback
            t_total = time.perf_counter() - t_start
            return {
                "sample_id": sample_id,
                "classical_top1": None,
                "candidates": [],
                "final_prediction": {
                    "x": None,
                    "y": None,
                    "rotation": None,
                    "scale": None,
                    "selected_candidate_rank": None
                },
                "elapsed_classical_s": round(elapsed_classical, 4),
                "elapsed_dl_s": 0.0,
                "elapsed_total_s": round(t_total, 4)
            }

        # Find classical top-1 candidate (rank_before_dl == 1)
        classical_top1 = None
        for c in candidates:
            if c["rank_before_dl"] == 1:
                classical_top1 = {
                    "x": c["x"],
                    "y": c["y"],
                    "rotation": c["rotation"],
                    "scale": c["scale"],
                    "classical_score": c["classical_score"]
                }
                break

        # 2. DL Verification
        t_dl_start = time.perf_counter()
        
        # Preprocess reference patch
        ref_tensor = self.patch_extractor.preprocess_patch(reference)
        
        # Extract and preprocess candidate patches
        cand_tensors = []
        for c in candidates:
            patch = self.patch_extractor.extract_candidate_patch(
                search, c["x"], c["y"], c["rotation"], c["scale"] / self.zoom_ratio
            )
            cand_tensors.append(self.patch_extractor.preprocess_patch(patch))
            
        # Run batched DL inference
        dl_results = self.reranker.verify_candidates(ref_tensor, cand_tensors)
        t_dl_end = time.perf_counter()
        elapsed_dl = t_dl_end - t_dl_start

        # 3. Score Normalization & Fusion
        class_scores = [c["classical_score"] for c in candidates]
        min_s = min(class_scores)
        max_s = max(class_scores)
        diff_s = max_s - min_s

        for idx, c in enumerate(candidates):
            # Normalize classical score
            if diff_s > 1e-6:
                norm_score = (c["classical_score"] - min_s) / diff_s
            else:
                norm_score = 1.0
                
            prob, sim = dl_results[idx]
            
            c["match_probability"] = prob
            c["embedding_similarity"] = sim
            
            # Compute fusion score based on mode
            if ranking_mode == "classical":
                c["hybrid_score"] = norm_score
            elif ranking_mode == "dl":
                c["hybrid_score"] = prob
            else:  # "hybrid"
                c["hybrid_score"] = norm_score + 0.01 * prob

        # Sort candidates based on fusion score descending
        sorted_candidates = sorted(candidates, key=lambda x: x["hybrid_score"], reverse=True)

        # Final prediction is the highest ranked candidate
        best_cand = sorted_candidates[0]

        # High-resolution coordinate descent pose refinement on final prediction
        refined_rot = best_cand["rotation"]
        refined_scale = best_cand["scale"]

        def ncc(im1, im2):
            im1_f = im1.astype(float) - np.mean(im1)
            im2_f = im2.astype(float) - np.mean(im2)
            denom = np.linalg.norm(im1_f) * np.linalg.norm(im2_f)
            if denom < 1e-8:
                return -1.0
            return np.sum(im1_f * im2_f) / denom

        # Iteration 1: Optimize rotation
        best_val = -1.0
        rot_grid = np.linspace(best_cand["rotation"] - 1.0, best_cand["rotation"] + 1.0, 15)
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
        drift_scale_grid = np.linspace(drift_s_center - 0.015, drift_s_center + 0.015, 11)
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
        rot_grid_fine = np.linspace(refined_rot - 0.2, refined_rot + 0.2, 5)
        for r in rot_grid_fine:
            patch = self.patch_extractor.extract_candidate_patch(
                search, best_cand["x"], best_cand["y"], r, refined_scale
            )
            val = ncc(reference, patch)
            if val > best_val:
                best_val = val
                refined_rot = r

        best_cand["rotation"] = float(refined_rot)
        best_cand["scale"] = float(refined_scale)

        final_prediction = {
            "x": best_cand["x"],
            "y": best_cand["y"],
            "rotation": best_cand["rotation"],
            "scale": best_cand["scale"],
            "selected_candidate_rank": best_cand["rank_before_dl"]
        }

        t_total = time.perf_counter() - t_start

        return {
            "sample_id": sample_id,
            "classical_top1": classical_top1,
            "candidates": sorted_candidates,
            "final_prediction": final_prediction,
            "elapsed_classical_s": round(elapsed_classical, 4),
            "elapsed_dl_s": round(elapsed_dl, 4),
            "elapsed_total_s": round(t_total, 4)
        }
