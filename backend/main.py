"""
DriftSense — FastAPI backend.
Presentation layer around the frozen classical NCC matcher + pose refinement + drift recovery pipeline.
"""
import os
import io
import csv
import json
import math
import time
import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    HealthResponse,
    AnalysisResponse,
    DemoResponse,
    CsvBatchResponse,
    CsvBatchSummary,
    PairEvaluationResult,
    AccuracyThreshold,
    BucketCount,
)
from backend.services.driftsense_service import DriftSenseService

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DriftSense API",
    description="Sensorless Wafer Navigation Drift Detection and Stage Coordinate Recovery",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Singleton service — matcher is warm after first request
# ---------------------------------------------------------------------------
service = DriftSenseService(config_path="configs/final_system_config.json")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_SAMPLE_ID = "sample_000"
DEMO_SAMPLE_DIR = PROJECT_ROOT / "data" / "phase2_test_data" / DEMO_SAMPLE_ID


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_grayscale(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")
    return img


def _img_to_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _find_column(headers: list, candidates: list) -> str:
    headers_clean = [h.strip().strip("\ufeff\"'").lower() for h in headers]
    for cand in candidates:
        cand_clean = cand.lower()
        if cand_clean in headers_clean:
            idx = headers_clean.index(cand_clean)
            return headers[idx]
    return ""


def _resolve_image_path(p_str: str) -> str:
    """Robust path resolution for external CSV files."""
    p_clean = p_str.strip().strip("\"'").replace("\\", "/")
    
    candidates = [
        Path(p_clean),
        PROJECT_ROOT / p_clean,
        PROJECT_ROOT / p_clean.lstrip("/"),
        PROJECT_ROOT / "data" / p_clean,
        PROJECT_ROOT / "data" / p_clean.lstrip("/"),
        PROJECT_ROOT / "data" / Path(p_clean).name,
    ]
    
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return str(cand.resolve())
            
    return p_clean


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="online", system="DriftSense")


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze(
    reference: UploadFile = File(...),
    search: UploadFile = File(...),
    expected_x: float = Form(...),
    expected_y: float = Form(...),
):
    ref_bytes = await reference.read()
    srch_bytes = await search.read()

    ref_img = _read_grayscale(ref_bytes)
    srch_img = _read_grayscale(srch_bytes)

    result = service.run_analysis(ref_img, srch_img, expected_x, expected_y)
    return AnalysisResponse(**result)


@app.get("/api/demo", response_model=DemoResponse)
def demo():
    ref_path = DEMO_SAMPLE_DIR / "reference_image.png"
    srch_path = DEMO_SAMPLE_DIR / "search_image.png"
    gt_path = DEMO_SAMPLE_DIR / "ground_truth.json"

    if not ref_path.exists() or not srch_path.exists() or not gt_path.exists():
        raise HTTPException(status_code=500, detail=f"Demo sample {DEMO_SAMPLE_ID} data not found.")

    ref_img = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
    srch_img = cv2.imread(str(srch_path), cv2.IMREAD_GRAYSCALE)

    with open(gt_path, "r") as f:
        gt = json.load(f)

    # Offset expected coordinates to simulate a visible MINOR_DRIFT scenario
    expected_x = gt["true_x"] + 2.1
    expected_y = gt["true_y"] - 1.5

    result = service.run_analysis(ref_img, srch_img, expected_x, expected_y)

    return DemoResponse(
        analysis=AnalysisResponse(**result),
        reference_image_b64=_img_to_b64(ref_path),
        search_image_b64=_img_to_b64(srch_path),
        sample_id=DEMO_SAMPLE_ID,
    )


@app.post("/api/analyze-csv", response_model=CsvBatchResponse)
async def analyze_csv(file: UploadFile = File(...)):
    contents = await file.read()
    text = contents.decode("utf-8-sig", errors="ignore")

    # Detect delimiter automatically (comma, semicolon, tab)
    delimiter = ","
    if "\t" in text.splitlines()[0]:
        delimiter = "\t"
    elif ";" in text.splitlines()[0] and "," not in text.splitlines()[0]:
        delimiter = ";"

    f = io.StringIO(text)
    reader = csv.DictReader(f, delimiter=delimiter)
    headers = reader.fieldnames or []

    # Comprehensive header synonyms
    srch_synonyms = [
        "wide search image path", "search_image_path", "search_path", "search", "search_image",
        "search_img", "search_file", "searchimage", "searchpath", "wide_search", "image_search",
        "search_image_file", "search_filename", "search_name"
    ]
    ref_synonyms = [
        "reference image path", "reference_image_path", "ref_path", "reference", "ref_image",
        "ref_img", "ref_file", "referenceimage", "referencepath", "ref", "template_image",
        "template_path", "template", "ref_image_file", "reference_filename", "ref_name"
    ]
    gtx_synonyms = ["gtx", "gt_x", "true_x", "x_gt", "target_x", "x", "x_center", "center_x", "gt_center_x", "true_center_x"]
    gty_synonyms = ["gty", "gt_y", "true_y", "y_gt", "target_y", "y", "y_center", "center_y", "gt_center_y", "true_center_y"]

    col_srch = _find_column(headers, srch_synonyms)
    col_ref = _find_column(headers, ref_synonyms)
    col_gtx = _find_column(headers, gtx_synonyms)
    col_gty = _find_column(headers, gty_synonyms)

    if not (col_srch and col_ref):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid CSV format. Found headers: {headers}. CSV must contain search image path and reference image path columns.",
        )

    rows = []
    for r in reader:
        item = {
            "search_path": r[col_srch].strip(),
            "ref_path": r[col_ref].strip(),
        }
        if col_gtx and col_gty and r.get(col_gtx) and r.get(col_gty):
            try:
                item["gt_x"] = float(r[col_gtx].strip())
                item["gt_y"] = float(r[col_gty].strip())
            except ValueError:
                pass
        rows.append(item)

    if not rows:
        raise HTTPException(status_code=400, detail="CSV file contains no data rows.")

    results = []
    total_time = 0.0
    errors = []
    missing_paths = []

    for i, item in enumerate(rows):
        srch_path = item["search_path"]
        ref_path = item["ref_path"]

        resolved_srch = _resolve_image_path(srch_path)
        resolved_ref = _resolve_image_path(ref_path)

        if not os.path.exists(resolved_srch) or not os.path.exists(resolved_ref):
            missing_paths.append((srch_path, ref_path))
            continue

        srch_img = cv2.imread(resolved_srch, cv2.IMREAD_GRAYSCALE)
        ref_img = cv2.imread(resolved_ref, cv2.IMREAD_GRAYSCALE)

        if srch_img is None or ref_img is None:
            missing_paths.append((srch_path, ref_path))
            continue

        t0 = time.perf_counter()
        pred = service.matcher.match(ref_img, srch_img)
        t_elapsed = time.perf_counter() - t0
        total_time += t_elapsed

        pred_x = pred["predicted_x"]
        pred_y = pred["predicted_y"]

        res_item = PairEvaluationResult(
            index=i + 1,
            search_image_path=srch_path,
            reference_image_path=ref_path,
            detected_x=round(pred_x, 4) if pred_x is not None else None,
            detected_y=round(pred_y, 4) if pred_y is not None else None,
            gt_x=item.get("gt_x"),
            gt_y=item.get("gt_y"),
            loc_error=None,
            rotation=round(pred["predicted_rotation"], 4) if pred["predicted_rotation"] is not None else None,
            scale=round(pred["predicted_scale"], 4) if pred["predicted_scale"] is not None else None,
            confidence=round(pred["confidence_score"], 4),
            elapsed_s=round(t_elapsed, 4),
        )

        if "gt_x" in item and "gt_y" in item:
            gt_x = item["gt_x"]
            gt_y = item["gt_y"]
            if pred_x is not None and pred_y is not None:
                loc_err = math.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
            else:
                loc_err = 999.0
            res_item.loc_error = round(loc_err, 4)
            errors.append(loc_err)

        results.append(res_item)

    N = len(results)
    if N == 0:
        sample_missing = missing_paths[0] if missing_paths else ("unknown", "unknown")
        raise HTTPException(
            status_code=400,
            detail=f"Could not locate image files referenced in CSV on server disk. Example missing pair: '{sample_missing[0]}' and '{sample_missing[1]}'. Ensure image files exist locally in the project directory.",
        )

    mean_err = round(float(np.mean(errors)), 4) if errors else None
    median_err = round(float(np.median(errors)), 4) if errors else None
    avg_time = round(total_time / N, 4)

    # Accuracy thresholds
    acc_thresholds = []
    for thresh in [1.0, 2.0, 3.0, 4.0, 5.0]:
        c_cnt = sum(1 for e in errors if e <= thresh) if errors else 0
        f_cnt = N - c_cnt
        pct = round((c_cnt / N) * 100.0, 2) if errors else 0.0
        acc_thresholds.append(
            AccuracyThreshold(
                threshold_px=thresh,
                correct_count=c_cnt,
                failed_count=f_cnt,
                accuracy_pct=pct,
            )
        )

    # Confusion matrix buckets
    b_0_1 = sum(1 for e in errors if e <= 1.0) if errors else 0
    b_1_2 = sum(1 for e in errors if 1.0 < e <= 2.0) if errors else 0
    b_2_3 = sum(1 for e in errors if 2.0 < e <= 3.0) if errors else 0
    b_3_4 = sum(1 for e in errors if 3.0 < e <= 4.0) if errors else 0
    b_4_5 = sum(1 for e in errors if 4.0 < e <= 5.0) if errors else 0
    b_gt5 = sum(1 for e in errors if e > 5.0) if errors else 0

    confusion_matrix = [
        BucketCount(bucket="[0.0, 1.0] px", count=b_0_1, percentage=round((b_0_1 / N) * 100.0, 2) if errors else 0.0),
        BucketCount(bucket="(1.0, 2.0] px", count=b_1_2, percentage=round((b_1_2 / N) * 100.0, 2) if errors else 0.0),
        BucketCount(bucket="(2.0, 3.0] px", count=b_2_3, percentage=round((b_2_3 / N) * 100.0, 2) if errors else 0.0),
        BucketCount(bucket="(3.0, 4.0] px", count=b_3_4, percentage=round((b_3_4 / N) * 100.0, 2) if errors else 0.0),
        BucketCount(bucket="(4.0, 5.0] px", count=b_4_5, percentage=round((b_4_5 / N) * 100.0, 2) if errors else 0.0),
        BucketCount(bucket="> 5.0 px (Outlier)", count=b_gt5, percentage=round((b_gt5 / N) * 100.0, 2) if errors else 0.0),
    ]

    summary = CsvBatchSummary(
        total_pairs=N,
        mean_loc_error=mean_err,
        median_loc_error=median_err,
        avg_inference_time_s=avg_time,
        accuracy_breakdown=acc_thresholds,
        confusion_matrix=confusion_matrix,
    )

    return CsvBatchResponse(summary=summary, results=results)
