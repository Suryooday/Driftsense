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
  search_width: number;
  search_height: number;
  found?: number;
}


export interface DemoResult {
  analysis: AnalysisResult;
  reference_image_b64: string;
  search_image_b64: string;
  sample_id: string;
}

// --- CSV Batch Evaluation Types ---

export interface PairEvaluationResult {
  index: number;
  pair_id?: string;
  search_image_path: string;
  reference_image_path: string;
  detected_x?: number;
  detected_y?: number;
  gt_x?: number;
  gt_y?: number;
  loc_error?: number;
  rotation?: number;
  scale?: number;
  confidence: number;
  found?: number;
  status?: string;
  elapsed_s: number;
}

export interface AccuracyThreshold {
  threshold_px: number;
  correct_count: number;
  failed_count: number;
  accuracy_pct: number;
}

export interface BucketCount {
  bucket: string;
  count: number;
  percentage: number;
}

export interface CsvBatchSummary {
  total_pairs: number;
  mean_loc_error?: number;
  median_loc_error?: number;
  avg_inference_time_s: number;
  accuracy_breakdown: AccuracyThreshold[];
  confusion_matrix: BucketCount[];
}

export interface CsvBatchResponse {
  summary: CsvBatchSummary;
  results: PairEvaluationResult[];
}
