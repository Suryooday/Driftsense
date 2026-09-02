import os
import sys
import math
import numpy as np
import cv2

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.final_system import FinalSystemMatcher, ncc
from scratch.test_periodicity_matching import GlobalSearchMatcher

def paste_template(search_img, template_patch, cx, cy):
    h, w = template_patch.shape
    y0 = int(cy - h // 2)
    x0 = int(cx - w // 2)
    y1 = y0 + h
    x1 = x0 + w
    
    # Simple blend/paste
    search_img[y0:y1, x0:x1] = template_patch
    return search_img

def main():
    print("======================================================================")
    print("RUNNING MULTI-INSTANCE PASTE TEST")
    print("======================================================================")
    
    # 1. Create a dummy reference template (e.g. a small square ring structure)
    ref_size = 256
    ref_img = np.full((ref_size, ref_size), 50, dtype=np.uint8)
    # Draw a bright ring
    cv2.circle(ref_img, (ref_size//2, ref_size//2), 60, 200, 16)
    cv2.circle(ref_img, (ref_size//2, ref_size//2), 20, 255, -1)
    
    # 2. Create a clean search image (plain background)
    search_size = 1000
    search_img_clean = np.full((search_size, search_size), 50, dtype=np.uint8)
    
    # Downscale template by 10x to represent 10x relative zoom in search image space
    ref_downscaled = cv2.resize(ref_img, (25, 25), interpolation=cv2.INTER_AREA)
    
    # 3. Paste the exact template at three different places:
    # - Copy 1 (True Center copy): placed near expected center (500, 500) -> e.g. at (503, 498)
    # - Copy 2 (Top-Left Duplicate): placed at (250, 250)
    # - Copy 3 (Bottom-Right Duplicate): placed at (750, 750)
    search_img_clean = paste_template(search_img_clean, ref_downscaled, 503, 498)
    search_img_clean = paste_template(search_img_clean, ref_downscaled, 250, 250)
    search_img_clean = paste_template(search_img_clean, ref_downscaled, 750, 750)
    
    # 4. Add some Gaussian noise to simulate realistic imaging
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 15, search_img_clean.shape).astype(np.float32)
    search_img = np.clip(search_img_clean.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # Initialize matchers
    gated_matcher = FinalSystemMatcher()
    global_matcher = GlobalSearchMatcher()
    
    # Run Global Search (Full 1000x1000 search space)
    print("\nRunning Global Search Matcher (No expected center crop constraint)...")
    res_global = global_matcher.match(ref_img, search_img)
    print(f"Global Match Result: Found = {res_global['found']}")
    print(f"Detected Coordinates: ({res_global['predicted_x']:.2f}, {res_global['predicted_y']:.2f})")
    print(f"Confidence Score (NCC): {res_global['confidence_score']:.4f}")
    
    # Run Gated Search (Our final system: crops to 100x100 around center)
    print("\nRunning Gated Search Matcher (Expected drift crop constraint enabled)...")
    res_gated = gated_matcher.match(ref_img, search_img)
    print(f"Gated Match Result: Found = {res_gated['found']}")
    print(f"Detected Coordinates: ({res_gated['predicted_x']:.2f}, {res_gated['predicted_y']:.2f})")
    print(f"Confidence Score (NCC): {res_gated['confidence_score']:.4f}")
    
    print("\n======================================================================")
    print("ANALYSIS:")
    print("======================================================================")
    print("1. Expected coordinate of stage was (500, 500).")
    print("2. True placed instance was at (503, 498) (drift of 3.6 pixels).")
    print("3. Duplicate identical patterns were present at (250, 250) and (750, 750).")
    print("\nDid Global Search find a duplicate? ")
    # Check if global search locked onto a duplicate
    dist_to_center_copy = math.sqrt((res_global['predicted_x'] - 503)**2 + (res_global['predicted_y'] - 498)**2)
    if dist_to_center_copy > 10.0:
        print("-> YES! Global search got confused by identical templates and locked onto a duplicate copy far from center.")
    else:
        print("-> NO. It managed to find the center copy.")
        
    print("\nDid Gated Search find the correct center copy? ")
    dist_to_gated_copy = math.sqrt((res_gated['predicted_x'] - 503)**2 + (res_gated['predicted_y'] - 498)**2)
    if dist_to_gated_copy <= 5.0:
        print("-> YES! Gated search correctly ignored all duplicate copies outside the drift zone and aligned successfully.")
    else:
        print("-> NO. It failed.")
    print("======================================================================")

if __name__ == "__main__":
    main()
