"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Crosshair, Upload } from "lucide-react";
import { useRef } from "react";

interface AnalysisControlsProps {
  expectedX: string;
  expectedY: string;
  onExpectedXChange: (v: string) => void;
  onExpectedYChange: (v: string) => void;
  onAnalyze: () => void;
  onReferenceUpload: (f: File) => void;
  onSearchUpload: (f: File) => void;
  isLoading: boolean;
  hasImages: boolean;
}

export function AnalysisControls({
  expectedX,
  expectedY,
  onExpectedXChange,
  onExpectedYChange,
  onAnalyze,
  onReferenceUpload,
  onSearchUpload,
  isLoading,
  hasImages,
}: AnalysisControlsProps) {
  const refInput = useRef<HTMLInputElement>(null);
  const srchInput = useRef<HTMLInputElement>(null);

  return (
    <div className="border-t border-gray-100 bg-white py-8">
      <div className="mx-auto max-w-7xl px-8">
        <div className="flex flex-wrap items-end gap-8">
          {/* Upload controls */}
          <div className="flex gap-3">
            <div>
              <label className="mb-1.5 block text-[11px] font-semibold uppercase tracking-widest text-gray-400">
                Reference Image
              </label>
              <input
                ref={refInput}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) onReferenceUpload(f);
                }}
              />
              <Button
                variant="outline"
                size="sm"
                className="gap-1.5 text-xs"
                onClick={() => refInput.current?.click()}
              >
                <Upload className="h-3 w-3" /> Upload
              </Button>
            </div>
            <div>
              <label className="mb-1.5 block text-[11px] font-semibold uppercase tracking-widest text-gray-400">
                Search Image
              </label>
              <input
                ref={srchInput}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) onSearchUpload(f);
                }}
              />
              <Button
                variant="outline"
                size="sm"
                className="gap-1.5 text-xs"
                onClick={() => srchInput.current?.click()}
              >
                <Upload className="h-3 w-3" /> Upload
              </Button>
            </div>
          </div>

          {/* Coordinate inputs */}
          <div>
            <label className="mb-1.5 block text-[11px] font-semibold uppercase tracking-widest text-gray-400">
              Expected Target Coordinates
            </label>
            <div className="flex gap-2">
              <div className="flex items-center gap-1.5">
                <span className="text-xs font-medium text-gray-500">X</span>
                <Input
                  type="number"
                  step="0.01"
                  value={expectedX}
                  onChange={(e) => onExpectedXChange(e.target.value)}
                  className="h-8 w-28 font-mono text-xs"
                  placeholder="0.00"
                />
              </div>
              <div className="flex items-center gap-1.5">
                <span className="text-xs font-medium text-gray-500">Y</span>
                <Input
                  type="number"
                  step="0.01"
                  value={expectedY}
                  onChange={(e) => onExpectedYChange(e.target.value)}
                  className="h-8 w-28 font-mono text-xs"
                  placeholder="0.00"
                />
              </div>
            </div>
          </div>

          {/* Analyze button */}
          <Button
            onClick={onAnalyze}
            disabled={isLoading || !hasImages}
            className="gap-2 bg-[#1E3A5F] text-white hover:bg-[#152d4a] text-xs font-semibold tracking-wider h-8 px-6"
          >
            <Crosshair className="h-3.5 w-3.5" />
            ANALYZE ALIGNMENT
          </Button>
        </div>
      </div>
    </div>
  );
}
