"use client";

import type { AnalysisResult } from "@/types/analysis";
import { CheckCircle2, AlertTriangle, XCircle, Clock, Compass, ZoomIn, Crosshair, ArrowUpRight } from "lucide-react";

interface AlignmentResultsProps {
  result: AnalysisResult;
}

function MetricCard({
  icon: Icon,
  label,
  children,
  badge,
}: {
  icon?: any;
  label: string;
  children: React.ReactNode;
  badge?: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-gray-100 bg-gray-50/50 p-4 transition-all hover:border-gray-200 hover:bg-white hover:shadow-sm">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5 text-gray-400">
          {Icon && <Icon className="h-3.5 w-3.5 text-[#1E3A5F]" />}
          <p className="text-[11px] font-semibold uppercase tracking-wider text-gray-500">
            {label}
          </p>
        </div>
        {badge}
      </div>
      <div className="mt-2.5">{children}</div>
    </div>
  );
}

export function AlignmentResults({ result }: AlignmentResultsProps) {
  const isFound = result.found !== 0 && result.drift.status !== "MATCH_FAILED";
  const relX = result.detected.x - result.search_width / 2;
  const relY = result.detected.y - result.search_height / 2;

  return (
    <section className="animate-fade-in border-t border-gray-100 bg-white py-10">
      <div className="mx-auto max-w-7xl px-8">
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h3 className="text-xl font-light tracking-tight text-[#1A1A2E]">
              Alignment & Navigation Telemetry
            </h3>
            <p className="mt-1 text-xs text-gray-400">
              Live stage coordinate recovery and multi-scale wafer pose estimation metrics.
            </p>
          </div>

          {/* Status Badge */}
          <div className="flex items-center gap-2">
            {result.drift.status === "ALIGNED" ? (
              <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-700 border border-emerald-200">
                <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />
                STAGE ALIGNED (≤ 1.0 px)
              </span>
            ) : result.drift.status === "MINOR_DRIFT" ? (
              <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-50 px-3 py-1 text-xs font-medium text-amber-700 border border-amber-200">
                <AlertTriangle className="h-3.5 w-3.5 text-amber-600" />
                MINOR DRIFT (≤ 5.0 px)
              </span>
            ) : result.drift.status === "SIGNIFICANT_DRIFT" ? (
              <span className="inline-flex items-center gap-1.5 rounded-full bg-rose-50 px-3 py-1 text-xs font-medium text-rose-700 border border-rose-200">
                <AlertTriangle className="h-3.5 w-3.5 text-rose-600" />
                SIGNIFICANT DRIFT ({result.drift.magnitude.toFixed(2)} px)
              </span>
            ) : (
              <span className="inline-flex items-center gap-1.5 rounded-full bg-gray-100 px-3 py-1 text-xs font-medium text-gray-700 border border-gray-200">
                <XCircle className="h-3.5 w-3.5 text-gray-500" />
                REFERENCE ABSENT / REJECTED
              </span>
            )}
          </div>
        </div>

        {/* 8-Metric Comprehensive Telemetry Grid */}
        <div className="grid grid-cols-4 gap-4">
          {/* 1. Detected Target Center */}
          <MetricCard icon={Crosshair} label="Detected Target (Absolute)">
            {isFound ? (
              <div>
                <p className="font-mono text-lg font-semibold text-[#1A1A2E]">
                  X: {result.detected.x.toFixed(2)} <span className="text-xs text-gray-400">px</span>
                </p>
                <p className="font-mono text-lg font-semibold text-[#1A1A2E]">
                  Y: {result.detected.y.toFixed(2)} <span className="text-xs text-gray-400">px</span>
                </p>
                <p className="mt-1 text-[11px] text-gray-400">
                  Rel: ({relX >= 0 ? "+" : ""}{relX.toFixed(1)}, {relY >= 0 ? "+" : ""}{relY.toFixed(1)}) px
                </p>
              </div>
            ) : (
              <p className="font-mono text-sm text-gray-400">Target Not Found</p>
            )}
          </MetricCard>

          {/* 2. Expected Inspection Center */}
          <MetricCard icon={Crosshair} label="Expected Site (Nominal)">
            <p className="font-mono text-lg font-semibold text-[#1A1A2E]">
              X: {result.expected.x.toFixed(2)} <span className="text-xs text-gray-400">px</span>
            </p>
            <p className="font-mono text-lg font-semibold text-[#1A1A2E]">
              Y: {result.expected.y.toFixed(2)} <span className="text-xs text-gray-400">px</span>
            </p>
            <p className="mt-1 text-[11px] text-gray-400">
              FOV: {result.search_width} × {result.search_height} px
            </p>
          </MetricCard>

          {/* 3. Navigation Drift Vector */}
          <MetricCard icon={ArrowUpRight} label="Navigation Drift">
            <p className="font-mono text-2xl font-bold text-[#1A1A2E]">
              {result.drift.magnitude.toFixed(2)} <span className="text-xs font-normal text-gray-400">px</span>
            </p>
            <p className="mt-1 font-mono text-xs text-gray-500">
              ΔX: {result.drift.dx >= 0 ? "+" : ""}{result.drift.dx.toFixed(2)} px | ΔY: {result.drift.dy >= 0 ? "+" : ""}{result.drift.dy.toFixed(2)} px
            </p>
          </MetricCard>

          {/* 4. Stage Correction */}
          <MetricCard icon={ArrowUpRight} label="Stage Correction Command">
            <p className="font-mono text-lg font-semibold text-[#1E3A5F]">
              MOVE X: {result.stage_correction.move_x >= 0 ? "+" : ""}{result.stage_correction.move_x.toFixed(2)} px
            </p>
            <p className="font-mono text-lg font-semibold text-[#1E3A5F]">
              MOVE Y: {result.stage_correction.move_y >= 0 ? "+" : ""}{result.stage_correction.move_y.toFixed(2)} px
            </p>
            <p className="mt-1 text-[11px] text-emerald-600 font-medium">
              Zero-sensor closed-loop recovery
            </p>
          </MetricCard>

          {/* 5. In-Plane Rotation */}
          <MetricCard icon={Compass} label="Stage In-Plane Rotation (θ)">
            <p className="font-mono text-2xl font-bold text-[#1A1A2E]">
              {result.pose.rotation >= 0 ? "+" : ""}{result.pose.rotation.toFixed(2)}°
            </p>
            <p className="mt-1 text-[11px] text-gray-400">
              Range: [-5.00°, +5.00°]
            </p>
          </MetricCard>

          {/* 6. Magnification Zoom Scale */}
          <MetricCard icon={ZoomIn} label="Zoom Magnification (z)">
            <p className="font-mono text-2xl font-bold text-[#1A1A2E]">
              {result.pose.scale.toFixed(3)}<span className="text-xs font-normal text-gray-400">×</span>
            </p>
            <p className="mt-1 text-[11px] text-gray-400">
              Continuous Scale z ∈ [8.0, 12.0]
            </p>
          </MetricCard>

          {/* 7. Normalized Cross-Correlation Confidence */}
          <MetricCard icon={CheckCircle2} label="Normalized Confidence (NCC)">
            <p className="font-mono text-2xl font-bold text-[#1A1A2E]">
              {result.confidence.ncc_score.toFixed(4)}
            </p>
            <p className="mt-1 text-[11px] text-gray-400">
              Rejection Threshold: τ = 0.550
            </p>
          </MetricCard>

          {/* 8. Inference Latency */}
          <MetricCard icon={Clock} label="Inference Execution Time">
            <p className="font-mono text-2xl font-bold text-emerald-700">
              {(result.inference_time_s * 1000).toFixed(1)} <span className="text-xs font-normal text-gray-400">ms</span>
            </p>
            <p className="mt-1 text-[11px] text-gray-400">
              {result.inference_time_s.toFixed(3)}s wall-clock latency
            </p>
          </MetricCard>
        </div>
      </div>
    </section>
  );
}
