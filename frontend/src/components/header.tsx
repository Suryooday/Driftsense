"use client";

import { Activity, Play } from "lucide-react";
import { Button } from "@/components/ui/button";

interface HeaderProps {
  isOnline: boolean;
  onRunDemo: () => void;
  isLoading: boolean;
}

export function Header({ isOnline, onRunDemo, isLoading }: HeaderProps) {
  return (
    <header className="border-b border-gray-200 bg-white">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-8 py-3">
        <div className="flex items-baseline gap-3">
          <h1 className="text-lg font-semibold tracking-tight text-[#1A1A2E]">
            DRIFTSENSE
          </h1>
          <span className="text-xs tracking-wide text-gray-400">
            Sensorless Wafer Navigation Recovery
          </span>
        </div>

        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <span>System Status</span>
            <span
              className={`inline-block h-2 w-2 rounded-full ${
                isOnline ? "bg-emerald-500" : "bg-gray-300"
              }`}
            />
            <span className="font-medium">
              {isOnline ? "ONLINE" : "OFFLINE"}
            </span>
          </div>

          <Button
            variant="outline"
            size="sm"
            onClick={onRunDemo}
            disabled={isLoading}
            className="gap-1.5 border-[#1E3A5F] text-[#1E3A5F] hover:bg-[#1E3A5F] hover:text-white text-xs"
          >
            <Play className="h-3 w-3" />
            Run Demo
          </Button>
        </div>
      </div>
    </header>
  );
}
