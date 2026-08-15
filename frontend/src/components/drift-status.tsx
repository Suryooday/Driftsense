"use client";

import type { DriftInfo } from "@/types/analysis";

interface DriftStatusProps {
  drift: DriftInfo;
}

const statusConfig = {
  ALIGNED: {
    color: "bg-emerald-500",
    textColor: "text-emerald-700",
    borderColor: "border-emerald-200",
    bgColor: "bg-emerald-50",
    label: "ALIGNED",
  },
  MINOR_DRIFT: {
    color: "bg-amber-500",
    textColor: "text-amber-700",
    borderColor: "border-amber-200",
    bgColor: "bg-amber-50",
    label: "MINOR DRIFT",
  },
  SIGNIFICANT_DRIFT: {
    color: "bg-red-500",
    textColor: "text-red-700",
    borderColor: "border-red-200",
    bgColor: "bg-red-50",
    label: "SIGNIFICANT DRIFT",
  },
  MATCH_FAILED: {
    color: "bg-gray-500",
    textColor: "text-gray-700",
    borderColor: "border-gray-200",
    bgColor: "bg-gray-50",
    label: "MATCH FAILED",
  },
};

export function DriftStatus({ drift }: DriftStatusProps) {
  const cfg = statusConfig[drift.status] || statusConfig.MATCH_FAILED;

  return (
    <section className="border-t border-gray-100 bg-white py-6">
      <div className="mx-auto max-w-7xl px-8">
        <div
          className={`inline-flex items-center gap-3 rounded border px-5 py-3 ${cfg.borderColor} ${cfg.bgColor}`}
        >
          <span className={`inline-block h-2.5 w-2.5 rounded-full ${cfg.color}`} />
          <span
            className={`text-sm font-semibold tracking-wider ${cfg.textColor}`}
          >
            {cfg.label}
          </span>
          <span className="text-xs text-gray-400">
            Δ {drift.magnitude.toFixed(2)} px
          </span>
        </div>
      </div>
    </section>
  );
}
