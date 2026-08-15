"""
DriftSense — FastAPI backend.
Presentation layer around the frozen classical NCC matcher + pose refinement + drift recovery pipeline.
"""
import os
import json
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
