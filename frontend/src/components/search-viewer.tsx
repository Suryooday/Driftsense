"use client";
 
import type { Coordinates } from "@/types/analysis";
 
interface SearchViewerProps {
  imageSrc: string | null;
  detected: Coordinates | null;
  isAnalyzing: boolean;
  searchWidth?: number;
  searchHeight?: number;
  matchFailed?: boolean;
}
 
export function SearchViewer({
  imageSrc,
  detected,
  isAnalyzing,
  searchWidth = 512,
  searchHeight = 512,
  matchFailed = false,
}: SearchViewerProps) {
  // Generate center-relative ticks: center is 0
  const getTicks = (limit: number) => {
    if (limit === 1000) {
      return [-500, -300, -100, 0, 100, 300, 500];
    }
    return [-256, -150, -50, 0, 50, 150, 256];
  };
 
  const xTicks = getTicks(searchWidth);
  const yTicks = getTicks(searchHeight);
 
  // Helper to convert relative tick values back to absolute percentage for visual positioning
  const getTickPct = (val: number, limit: number) => {
    const absVal = val + limit / 2;
    return (absVal / limit) * 100;
  };
 
  const showOverlay = detected && !isAnalyzing && !matchFailed;
 
  return (
    <div>
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-widest text-gray-400">
          Search Area
        </h3>
        <span className="font-mono text-[10px] text-gray-400">
          {searchWidth} × {searchHeight} px (Centered Origin)
        </span>
      </div>
 
      <div className="rounded border border-gray-200 bg-gray-50 p-2">
        {/* Top X-Axis Ruler */}
        <div className="mb-1.5 flex h-4 pl-6 pr-1 relative">
          <div className="relative w-full h-full">
            {xTicks.map((val) => (
              <div
                key={`top-${val}`}
                className="absolute top-0 flex flex-col items-center -translate-x-1/2"
                style={{ left: `${getTickPct(val, searchWidth)}%` }}
              >
                <span className="font-mono text-[9px] text-gray-400 select-none">
                  {val >= 0 ? "+" : ""}{val}
                </span>
                <div className="h-1 w-px bg-gray-300 mt-0.5" />
              </div>
            ))}
 
            {/* Indicator tick for detected X */}
            {showOverlay && (
              <div
                className="absolute top-0 flex flex-col items-center -translate-x-1/2 z-10 animate-fade-in"
                style={{ left: `${(detected.x / searchWidth) * 100}%` }}
              >
                <span className="font-mono text-[9px] font-semibold text-[#1E3A5F] bg-[#1E3A5F]/10 px-1 rounded select-none">
                  {detected.x - searchWidth / 2 >= 0 ? "+" : ""}
                  {(detected.x - searchWidth / 2).toFixed(0)}
                </span>
                <div className="h-1.5 w-0.5 bg-[#1E3A5F]" />
              </div>
            )}
          </div>
        </div>
 
        <div className="flex">
          {/* Left Y-Axis Ruler */}
          <div className="relative w-6 flex-shrink-0">
            {yTicks.map((val) => (
              <div
                key={`left-${val}`}
                className="absolute left-0 flex items-center -translate-y-1/2 w-full justify-end pr-1"
                style={{ top: `${getTickPct(val, searchHeight)}%` }}
              >
                <span className="font-mono text-[9px] text-gray-400 select-none">
                  {val >= 0 ? "+" : ""}{val}
                </span>
                <div className="h-px w-1 bg-gray-300 ml-0.5" />
              </div>
            ))}
 
            {/* Indicator tick for detected Y */}
            {showOverlay && (
              <div
                className="absolute left-0 flex items-center -translate-y-1/2 w-full justify-end pr-0.5 z-10 animate-fade-in"
                style={{ top: `${(detected.y / searchHeight) * 100}%` }}
              >
                <span className="font-mono text-[9px] font-semibold text-[#1E3A5F] bg-[#1E3A5F]/10 px-0.5 rounded select-none">
                  {detected.y - searchHeight / 2 >= 0 ? "+" : ""}
                  {(detected.y - searchHeight / 2).toFixed(0)}
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
                {showOverlay && (
                  <div className="pointer-events-none absolute inset-0 animate-fade-in">
                    {/* Crosshair */}
                    <svg className="absolute inset-0 h-full w-full">
                      {/* Convert pixel coords to percentage for responsive overlay */}
                      <line
                        x1={`${(detected.x / searchWidth) * 100}%`}
                        y1="0"
                        x2={`${(detected.x / searchWidth) * 100}%`}
                        y2="100%"
                        stroke="#1E3A5F"
                        strokeWidth="0.5"
                        strokeDasharray="4 4"
                        opacity="0.6"
                      />
                      <line
                        x1="0"
                        y1={`${(detected.y / searchHeight) * 100}%`}
                        x2="100%"
                        y2={`${(detected.y / searchHeight) * 100}%`}
                        stroke="#1E3A5F"
                        strokeWidth="0.5"
                        strokeDasharray="4 4"
                        opacity="0.6"
                      />
                      {/* Center circle */}
                      <circle
                        cx={`${(detected.x / searchWidth) * 100}%`}
                        cy={`${(detected.y / searchHeight) * 100}%`}
                        r="6"
                        fill="none"
                        stroke="#1E3A5F"
                        strokeWidth="1.5"
                      />
                      <circle
                        cx={`${(detected.x / searchWidth) * 100}%`}
                        cy={`${(detected.y / searchHeight) * 100}%`}
                        r="2"
                        fill="#1E3A5F"
                      />
                    </svg>
 
                    {/* Label */}
                    <div className="absolute bottom-3 left-3 rounded bg-[#1A1A2E]/85 px-2.5 py-1.5 backdrop-blur-sm shadow-sm">
                      <p className="text-[10px] font-semibold uppercase tracking-widest text-white/70">
                        Detected Target (Rel to Center)
                      </p>
                      <p className="mt-0.5 font-mono text-xs text-white">
                        X: {detected.x - searchWidth / 2 >= 0 ? "+" : ""}
                        {(detected.x - searchWidth / 2).toFixed(2)}&ensp;
                        Y: {detected.y - searchHeight / 2 >= 0 ? "+" : ""}
                        {(detected.y - searchHeight / 2).toFixed(2)}
                      </p>
                    </div>
                  </div>
                )}
 
                {/* Match failure visual cover */}
                {matchFailed && !isAnalyzing && (
                  <div className="absolute inset-0 flex items-center justify-center bg-red-950/15 backdrop-blur-[0.5px]">
                    <div className="rounded bg-red-900/90 px-4 py-2.5 text-center text-white shadow-md">
                      <p className="text-xs font-semibold uppercase tracking-widest">Alignment Failed</p>
                      <p className="mt-0.5 text-[10px] text-red-200">Reference pattern could not be located</p>
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


