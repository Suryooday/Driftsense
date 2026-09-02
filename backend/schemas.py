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



class DemoResponse(BaseModel):
    analysis: AnalysisResponse
    reference_image_b64: str
    search_image_b64: str
    sample_id: str


# --- CSV Batch Evaluation Schemas ---

class PairEvaluationResult(BaseModel):
    index: int
    search_image_path: str
    reference_image_path: str
    detected_x: Optional[float]
    detected_y: Optional[float]
    gt_x: Optional[float]
    gt_y: Optional[float]
    loc_error: Optional[float]
    rotation: Optional[float]
    scale: Optional[float]
    confidence: float
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
    mean_loc_error: Optional[float]
    median_loc_error: Optional[float]
    avg_inference_time_s: float
    accuracy_breakdown: List[AccuracyThreshold]
    confusion_matrix: List[BucketCount]


class CsvBatchResponse(BaseModel):
    summary: CsvBatchSummary
    results: List[PairEvaluationResult]
