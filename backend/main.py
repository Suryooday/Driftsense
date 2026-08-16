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
DEMO_SAMPLE_ID = "sample_010"
DEMO_SAMPLE_DIR = PROJECT_ROOT / "data" / DEMO_SAMPLE_ID


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
    headers_lower = [h.strip().lower() for h in headers]
    for cand in candidates:
        cand_lower = cand.lower()
        if cand_lower in headers_lower:
            idx = headers_lower.index(cand_lower)
            return headers[idx]
    return ""


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
    expected_x = gt["true_x"] + 3.5
    expected_y = gt["true_y"] - 2.0

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
    text = contents.decode("utf-8", errors="ignore")

    f = io.StringIO(text)
    reader = csv.DictReader(f)
    headers = reader.fieldnames or []

    col_srch = _find_column(headers, ["wide search image path", "search_image_path", "search_path", "search", "search_image"])
    col_ref = _find_column(headers, ["reference image path", "reference_image_path", "ref_path", "reference", "ref_image"])
    col_gtx = _find_column(headers, ["gtx", "gt_x", "true_x", "x_gt", "target_x"])
    col_gty = _find_column(headers, ["gty", "gt_y", "true_y", "y_gt", "target_y"])

    if not (col_srch and col_ref):
        raise HTTPException(
            status_code=400,
            detail=f"Missing image path columns. Headers found: {headers}. Expected search_image_path & reference_image_path.",
        )

    rows = []
    for r in reader:
        item = {
            "search_path": r[col_srch].strip(),
            "ref_path": r[col_ref].strip(),
        }
        if col_gtx and col_gty and r[col_gtx] and r[col_gty]:
            try:
                item["gt_x"] = float(r[col_gtx])
                item["gt_y"] = float(r[col_gty])
            except ValueError:
                pass
        rows.append(item)

    if not rows:
        raise HTTPException(status_code=400, detail="CSV file contains no data rows.")

    results = []
    total_time = 0.0
    errors = []

    for i, item in enumerate(rows):
        srch_path = item["search_path"]
        ref_path = item["ref_path"]

        # Resolve paths relative to PROJECT_ROOT if needed
        full_srch = srch_path if os.path.exists(srch_path) else str(PROJECT_ROOT / srch_path)
        full_ref = ref_path if os.path.exists(ref_path) else str(PROJECT_ROOT / ref_path)

        if not os.path.exists(full_srch) or not os.path.exists(full_ref):
            continue

        srch_img = cv2.imread(full_srch, cv2.IMREAD_GRAYSCALE)
        ref_img = cv2.imread(full_ref, cv2.IMREAD_GRAYSCALE)

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
        raise HTTPException(status_code=400, detail="Could not evaluate any valid image pairs from CSV.")

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
