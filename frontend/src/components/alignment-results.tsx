"use client";

import type { AnalysisResult } from "@/types/analysis";

interface AlignmentResultsProps {
  result: AnalysisResult;
}

function MetricCard({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="border-l-2 border-gray-100 pl-4">
      <p className="text-[11px] font-semibold uppercase tracking-widest text-gray-400">
        {label}
      </p>
      <div className="mt-1">{children}</div>
    </div>
  );
}

export function AlignmentResults({ result }: AlignmentResultsProps) {
  return (
    <section className="animate-fade-in border-t border-gray-100 bg-white py-10">
      <div className="mx-auto max-w-7xl px-8">
        <h3 className="mb-8 text-xl font-light tracking-tight text-[#1A1A2E]">
          Alignment Result
        </h3>

        <div className="grid grid-cols-5 gap-8">
          <MetricCard label="Detected Position">
            <p className="font-mono text-lg font-medium text-[#1A1A2E]">
              {result.detected.x.toFixed(2)}{" "}
              <span className="text-xs text-gray-400">px</span>
            </p>
            <p className="font-mono text-lg font-medium text-[#1A1A2E]">
              {result.detected.y.toFixed(2)}{" "}
              <span className="text-xs text-gray-400">px</span>
            </p>
          </MetricCard>

          <MetricCard label="Navigation Drift">
            <p className="font-mono text-2xl font-semibold text-[#1A1A2E]">
              {result.drift.magnitude.toFixed(2)}
            </p>
            <p className="text-xs text-gray-400">px</p>
          </MetricCard>

          <MetricCard label="Rotation">
            <p className="font-mono text-2xl font-semibold text-[#1A1A2E]">
              {result.pose.rotation.toFixed(2)}°
            </p>
          </MetricCard>

          <MetricCard label="Scale">
            <p className="font-mono text-2xl font-semibold text-[#1A1A2E]">
              {result.pose.scale.toFixed(3)}
            </p>
          </MetricCard>

          <MetricCard label="Match Confidence">
            <p className="font-mono text-2xl font-semibold text-[#1A1A2E]">
              {result.confidence.ncc_score.toFixed(3)}
            </p>
            <p className="text-xs text-gray-400">NCC</p>
          </MetricCard>
        </div>
      </div>
    </section>
  );
}
