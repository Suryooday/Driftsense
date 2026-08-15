"use client";

interface ReferenceViewerProps {
  imageSrc: string | null;
}

const ticks256 = [0, 50, 100, 150, 200, 250];

export function ReferenceViewer({ imageSrc }: ReferenceViewerProps) {
  return (
    <div>
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-widest text-gray-400">
          Reference Pattern
        </h3>
        <span className="font-mono text-[10px] text-gray-400">256 × 256 px</span>
      </div>

      <div className="rounded border border-gray-200 bg-gray-50 p-2">
        {/* Top X-Axis Ruler */}
        <div className="mb-1.5 flex h-4 pl-6 pr-1 relative">
          <div className="relative w-full h-full">
            {ticks256.map((val) => (
              <div
                key={`top-${val}`}
                className="absolute top-0 flex flex-col items-center -translate-x-1/2"
                style={{ left: `${(val / 256) * 100}%` }}
              >
                <span className="font-mono text-[9px] text-gray-400 select-none">
                  {val}
                </span>
                <div className="h-1 w-px bg-gray-300 mt-0.5" />
              </div>
            ))}
          </div>
        </div>

        <div className="flex">
          {/* Left Y-Axis Ruler */}
          <div className="relative w-6 flex-shrink-0">
            {ticks256.map((val) => (
              <div
                key={`left-${val}`}
                className="absolute left-0 flex items-center -translate-y-1/2 w-full justify-end pr-1"
                style={{ top: `${(val / 256) * 100}%` }}
              >
                <span className="font-mono text-[9px] text-gray-400 select-none">
                  {val}
                </span>
                <div className="h-px w-1 bg-gray-300 ml-0.5" />
              </div>
            ))}
          </div>

          {/* Main Image Canvas */}
          <div className="relative aspect-square w-full overflow-hidden rounded border border-gray-200 bg-white">
            {imageSrc ? (
              <img
                src={imageSrc}
                alt="Reference pattern"
                className="h-full w-full object-contain"
              />
            ) : (
              <div className="flex h-full items-center justify-center">
                <span className="text-xs text-gray-300">No image loaded</span>
              </div>
            )}
          </div>
        </div>
      </div>

      <p className="mt-2 text-[11px] text-gray-400">
        Reference Resolution&ensp;·&ensp;100× Inspection Pattern
      </p>
    </div>
  );
}
