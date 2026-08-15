"use client";

import type { Coordinates } from "@/types/analysis";

interface SearchViewerProps {
  imageSrc: string | null;
  detected: Coordinates | null;
  isAnalyzing: boolean;
}

const ticks512 = [0, 100, 200, 300, 400, 500];

export function SearchViewer({
  imageSrc,
  detected,
  isAnalyzing,
}: SearchViewerProps) {
  return (
    <div>
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-widest text-gray-400">
          Search Area
        </h3>
        <span className="font-mono text-[10px] text-gray-400">512 × 512 px</span>
      </div>

      <div className="rounded border border-gray-200 bg-gray-50 p-2">
        {/* Top X-Axis Ruler */}
        <div className="mb-1.5 flex h-4 pl-6 pr-1 relative">
          <div className="relative w-full h-full">
            {ticks512.map((val) => (
              <div
                key={`top-${val}`}
                className="absolute top-0 flex flex-col items-center -translate-x-1/2"
                style={{ left: `${(val / 512) * 100}%` }}
              >
                <span className="font-mono text-[9px] text-gray-400 select-none">
                  {val}
                </span>
                <div className="h-1 w-px bg-gray-300 mt-0.5" />
              </div>
            ))}

            {/* Indicator tick for detected X */}
            {detected && !isAnalyzing && (
              <div
                className="absolute top-0 flex flex-col items-center -translate-x-1/2 z-10 animate-fade-in"
                style={{ left: `${(detected.x / 512) * 100}%` }}
              >
                <span className="font-mono text-[9px] font-semibold text-[#1E3A5F] bg-[#1E3A5F]/10 px-1 rounded select-none">
                  {detected.x.toFixed(0)}
                </span>
                <div className="h-1.5 w-0.5 bg-[#1E3A5F]" />
              </div>
            )}
          </div>
        </div>

        <div className="flex">
          {/* Left Y-Axis Ruler */}
          <div className="relative w-6 flex-shrink-0">
            {ticks512.map((val) => (
              <div
                key={`left-${val}`}
                className="absolute left-0 flex items-center -translate-y-1/2 w-full justify-end pr-1"
                style={{ top: `${(val / 512) * 100}%` }}
              >
                <span className="font-mono text-[9px] text-gray-400 select-none">
                  {val}
                </span>
                <div className="h-px w-1 bg-gray-300 ml-0.5" />
              </div>
            ))}

            {/* Indicator tick for detected Y */}
            {detected && !isAnalyzing && (
              <div
                className="absolute left-0 flex items-center -translate-y-1/2 w-full justify-end pr-0.5 z-10 animate-fade-in"
                style={{ top: `${(detected.y / 512) * 100}%` }}
              >
                <span className="font-mono text-[9px] font-semibold text-[#1E3A5F] bg-[#1E3A5F]/10 px-0.5 rounded select-none">
                  {detected.y.toFixed(0)}
                </span>
                <div className="h-0.5 w-1.5 bg-[#1E3A5F]" />
              </div>
            )}
          </div>

          {/* Main Search Canvas */}
          <div className="relative aspect-square w-full overflow-hidden rounded border border-gray-200 bg-white">
            {imageSrc ? (
              <>
                <img
                  src={imageSrc}
                  alt="Search area"
                  className="h-full w-full object-contain"
                />

                {/* Scanning line animation during analysis */}
                {isAnalyzing && (
                  <div className="pointer-events-none absolute inset-0">
                    <div className="absolute left-0 right-0 h-px bg-[#1E3A5F]/60 animate-scan" />
                  </div>
                )}

                {/* Detected target overlay */}
                {detected && !isAnalyzing && (
                  <div className="pointer-events-none absolute inset-0 animate-fade-in">
                    {/* Crosshair */}
                    <svg className="absolute inset-0 h-full w-full">
                      {/* Convert pixel coords to percentage for responsive overlay */}
                      <line
                        x1={`${(detected.x / 512) * 100}%`}
                        y1="0"
                        x2={`${(detected.x / 512) * 100}%`}
                        y2="100%"
                        stroke="#1E3A5F"
                        strokeWidth="0.5"
                        strokeDasharray="4 4"
                        opacity="0.6"
                      />
                      <line
                        x1="0"
                        y1={`${(detected.y / 512) * 100}%`}
                        x2="100%"
                        y2={`${(detected.y / 512) * 100}%`}
                        stroke="#1E3A5F"
                        strokeWidth="0.5"
                        strokeDasharray="4 4"
                        opacity="0.6"
                      />
                      {/* Center circle */}
                      <circle
                        cx={`${(detected.x / 512) * 100}%`}
                        cy={`${(detected.y / 512) * 100}%`}
                        r="6"
                        fill="none"
                        stroke="#1E3A5F"
                        strokeWidth="1.5"
                      />
                      <circle
                        cx={`${(detected.x / 512) * 100}%`}
                        cy={`${(detected.y / 512) * 100}%`}
                        r="2"
                        fill="#1E3A5F"
                      />
                    </svg>

                    {/* Label */}
                    <div className="absolute bottom-3 left-3 rounded bg-[#1A1A2E]/85 px-2.5 py-1.5 backdrop-blur-sm shadow-sm">
                      <p className="text-[10px] font-semibold uppercase tracking-widest text-white/70">
                        Detected Target
                      </p>
                      <p className="mt-0.5 font-mono text-xs text-white">
                        X: {detected.x.toFixed(2)}&ensp; Y: {detected.y.toFixed(2)}
                      </p>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="flex h-full items-center justify-center">
                <span className="text-xs text-gray-300">No image loaded</span>
              </div>
            )}
          </div>
        </div>
      </div>

      <p className="mt-2 text-[11px] text-gray-400">
        Search Region&ensp;·&ensp;10× Wafer FOV
      </p>
    </div>
  );
}
