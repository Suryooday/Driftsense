"""
Pydantic response models for the DriftSense API.
"""
from pydantic import BaseModel
from typing import Optional


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


class DemoResponse(BaseModel):
    analysis: AnalysisResponse
    reference_image_b64: str
    search_image_b64: str
    sample_id: str
