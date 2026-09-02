"""
Pydantic response models for the DriftSense API.
"""
from pydantic import BaseModel
from typing import Optional, List


class HealthResponse(BaseModel):
    status: str
    system: str


class Coordinates(BaseModel):
    x: float
    y: float


class DriftInfo(BaseModel):
    dx: float
    dy: float
    magnitude: float
    status: str


class PoseInfo(BaseModel):
    rotation: float
    scale: float


class ConfidenceInfo(BaseModel):
    ncc_score: float


class StageCorrection(BaseModel):
    move_x: float
    move_y: float


class AnalysisResponse(BaseModel):
    expected: Coordinates
    detected: Coordinates
    drift: DriftInfo
    pose: PoseInfo
    confidence: ConfidenceInfo
    stage_correction: StageCorrection
    inference_time_s: float
    search_width: float
    search_height: float
    found: Optional[int] = 1


class DemoResponse(BaseModel):
    analysis: AnalysisResponse
    reference_image_b64: str
    search_image_b64: str
    sample_id: str


# --- CSV Batch Evaluation Schemas ---

class PairEvaluationResult(BaseModel):
    index: int
    pair_id: Optional[str] = None
    search_image_path: str
    reference_image_path: str
    detected_x: Optional[float] = None
    detected_y: Optional[float] = None
    gt_x: Optional[float] = None
    gt_y: Optional[float] = None
    loc_error: Optional[float] = None
    rotation: Optional[float] = None
    scale: Optional[float] = None
    confidence: float
    found: int = 1
    status: Optional[str] = None
    elapsed_s: float


class AccuracyThreshold(BaseModel):
    threshold_px: float
    correct_count: int
    failed_count: int
    accuracy_pct: float


class BucketCount(BaseModel):
    bucket: str
    count: int
    percentage: float


class CsvBatchSummary(BaseModel):
    total_pairs: int
    mean_loc_error: Optional[float] = None
    median_loc_error: Optional[float] = None
    avg_inference_time_s: float
    accuracy_breakdown: List[AccuracyThreshold]
    confusion_matrix: List[BucketCount]


class CsvBatchResponse(BaseModel):
    summary: CsvBatchSummary
    results: List[PairEvaluationResult]
