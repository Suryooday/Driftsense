"use client";

import { useEffect, useState } from "react";
import type { AnalysisResult } from "@/types/analysis";

interface StageRecoveryProps {
  result: AnalysisResult;
}

export function StageRecovery({ result }: StageRecoveryProps) {
  const [animProgress, setAnimProgress] = useState(0);

  useEffect(() => {
    setAnimProgress(0);
    const timer = setTimeout(() => setAnimProgress(1), 100);
    return () => clearTimeout(timer);
  }, [result]);

  const { stage_correction, drift } = result;

  // SVG dimensions
  const w = 400;
  const h = 200;
  const cx = w / 2;
  const cy = h / 2;

  // Scale the vector for visibility (cap at 60px visual)
  const mag = drift.magnitude;
  const scale = mag > 0 ? Math.min(60 / mag, 15) : 1;
  const vx = stage_correction.move_x * scale;
  const vy = stage_correction.move_y * scale;

  // Detected position (origin of arrow)
  const dx = cx - vx / 2;
  const dy = cy - vy / 2;
  // Expected position (tip of arrow)
  const ex = cx + vx / 2;
  const ey = cy + vy / 2;

  return (
    <section className="animate-fade-in border-t border-gray-100 bg-white py-10">
      <div className="mx-auto max-w-7xl px-8">
        <h3 className="mb-8 text-xl font-light tracking-tight text-[#1A1A2E]">
          Recommended Stage Correction
        </h3>

        <div className="flex items-start gap-16">
          {/* Vector visualization */}
          <div className="rounded border border-gray-200 bg-gray-50 p-2">
            <svg
              width={w}
              height={h}
              viewBox={`0 0 ${w} ${h}`}
              className="block"
            >
              {/* Grid lines */}
              <line
                x1={cx}
                y1="0"
                x2={cx}
                y2={h}
                stroke="#e5e7eb"
                strokeWidth="0.5"
              />
              <line
                x1="0"
                y1={cy}
                x2={w}
                y2={cy}
                stroke="#e5e7eb"
                strokeWidth="0.5"
              />

              {/* Correction arrow */}
              <line
                x1={dx}
                y1={dy}
                x2={dx + (ex - dx) * animProgress}
                y2={dy + (ey - dy) * animProgress}
                stroke="#1E3A5F"
                strokeWidth="2"
                markerEnd="url(#arrowhead)"
                className="transition-all duration-700 ease-out"
              />

              {/* Current position (detected) */}
              <circle cx={dx} cy={dy} r="4" fill="#DC2626" />
              <text
                x={dx}
                y={dy - 10}
                textAnchor="middle"
                className="fill-gray-500 text-[9px] font-medium"
              >
                CURRENT
              </text>

              {/* Target position (expected) */}
              <circle
                cx={ex}
                cy={ey}
                r="6"
                fill="none"
                stroke="#16A34A"
                strokeWidth="1.5"
                opacity={animProgress}
                className="transition-opacity duration-500 delay-500"
              />
              <circle
                cx={ex}
                cy={ey}
                r="2"
                fill="#16A34A"
                opacity={animProgress}
                className="transition-opacity duration-500 delay-500"
              />
              <text
                x={ex}
                y={ey + 16}
                textAnchor="middle"
                className="fill-gray-500 text-[9px] font-medium"
                opacity={animProgress}
              >
                TARGET
              </text>

              <defs>
                <marker
                  id="arrowhead"
                  markerWidth="8"
                  markerHeight="6"
                  refX="8"
                  refY="3"
                  orient="auto"
                >
                  <polygon
                    points="0 0, 8 3, 0 6"
                    fill="#1E3A5F"
                  />
                </marker>
              </defs>
            </svg>
          </div>

          {/* Correction values */}
          <div className="space-y-6">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-widest text-gray-400">
                Move X
              </p>
              <p className="mt-1 font-mono text-2xl font-semibold text-[#1A1A2E]">
                {stage_correction.move_x >= 0 ? "+" : ""}
                {stage_correction.move_x.toFixed(2)}{" "}
                <span className="text-sm font-normal text-gray-400">px</span>
              </p>
            </div>
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-widest text-gray-400">
                Move Y
              </p>
              <p className="mt-1 font-mono text-2xl font-semibold text-[#1A1A2E]">
                {stage_correction.move_y >= 0 ? "+" : ""}
                {stage_correction.move_y.toFixed(2)}{" "}
                <span className="text-sm font-normal text-gray-400">px</span>
              </p>
            </div>
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-widest text-gray-400">
                Total Correction
              </p>
              <p className="mt-1 font-mono text-2xl font-semibold text-[#1E3A5F]">
                {drift.magnitude.toFixed(2)}{" "}
                <span className="text-sm font-normal text-gray-400">px</span>
              </p>
            </div>
          </div>
        </div>

        <p className="mt-8 max-w-2xl text-xs leading-relaxed text-gray-400">
          DriftSense converts the detected pixel offset into a corrective stage
          movement vector, allowing the inspection system to recover the
          intended target location. Physical stage displacement requires
          external pixel-to-stage calibration.
        </p>
      </div>
    </section>
  );
}
