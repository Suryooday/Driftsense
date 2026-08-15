export interface Coordinates {
  x: number;
  y: number;
}

export interface DriftInfo {
  dx: number;
  dy: number;
  magnitude: number;
  status: "ALIGNED" | "MINOR_DRIFT" | "SIGNIFICANT_DRIFT" | "MATCH_FAILED";
}

export interface PoseInfo {
  rotation: number;
  scale: number;
}

export interface ConfidenceInfo {
  ncc_score: number;
}

export interface StageCorrection {
  move_x: number;
  move_y: number;
}

export interface AnalysisResult {
  expected: Coordinates;
  detected: Coordinates;
  drift: DriftInfo;
  pose: PoseInfo;
  confidence: ConfidenceInfo;
  stage_correction: StageCorrection;
  inference_time_s: number;
}

export interface DemoResult {
  analysis: AnalysisResult;
  reference_image_b64: string;
  search_image_b64: string;
  sample_id: string;
}
