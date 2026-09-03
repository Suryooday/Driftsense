"use client";

import { useState, useRef } from "react";
import type { CsvBatchResponse, PairEvaluationResult } from "@/types/analysis";
import { analyzeCsv } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { 
  Upload, 
  FileSpreadsheet, 
  CheckCircle2, 
  AlertCircle, 
  Search, 
  Download, 
  Compass, 
  ZoomIn, 
  Crosshair, 
  Clock,
  Filter
} from "lucide-react";
import { Input } from "@/components/ui/input";

export function CsvBatchViewer() {
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [batchData, setBatchData] = useState<CsvBatchResponse | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<"all" | "found" | "absent">("all");

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) {
      setFile(selected);
      setErrorMsg(null);
    }
  };

  const handleRunEvaluation = async (fileToRun?: File) => {
    const targetFile = fileToRun || file;
    if (!targetFile) return;
    setIsLoading(true);
    setErrorMsg(null);

    try {
      const res = await analyzeCsv(targetFile);
      setBatchData(res);
    } catch (err: any) {
      setErrorMsg(err.message || "Failed to analyze CSV dataset.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleLoadSample = async (samplePath: string, sampleName: string) => {
    try {
      setIsLoading(true);
      setErrorMsg(null);
      const res = await fetch(`/${samplePath}`);
      if (!res.ok) {
        throw new Error(`Could not load preset ${sampleName}`);
      }
      const text = await res.text();
      const loadedFile = new File([text], sampleName, { type: "text/csv" });
      setFile(loadedFile);
      await handleRunEvaluation(loadedFile);
    } catch (e: any) {
      // Fallback: create mock file with standard output pairs
      setErrorMsg(`Preset not directly fetchable via browser URL. Please select '${sampleName}' using the file picker.`);
      setIsLoading(false);
    }
  };

  // Export predictions.csv matching the exact evaluation contract
  const handleDownloadPredictions = () => {
    if (!batchData || !batchData.results.length) return;

    const headers = ["pair_id", "x", "y", "theta", "scale", "found", "score"];
    const rows = batchData.results.map((r) => [
      r.pair_id || `pair_${r.index}`,
      r.found !== 0 && r.detected_x !== null && r.detected_x !== undefined ? r.detected_x.toFixed(4) : "0.0000",
      r.found !== 0 && r.detected_y !== null && r.detected_y !== undefined ? r.detected_y.toFixed(4) : "0.0000",
      r.found !== 0 && r.rotation !== null && r.rotation !== undefined ? r.rotation.toFixed(4) : "0.0000",
      r.found !== 0 && r.scale !== null && r.scale !== undefined ? r.scale.toFixed(4) : "0.0000",
      r.found !== undefined ? r.found : 1,
      r.confidence.toFixed(4),
    ]);

    const csvContent = [
      headers.join(","),
      ...rows.map((row) => row.join(",")),
    ].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", "predictions.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Filter pairs
  const filteredResults = batchData
    ? batchData.results.filter((r) => {
        const matchesQuery =
          (r.pair_id && r.pair_id.toLowerCase().includes(searchQuery.toLowerCase())) ||
          r.search_image_path.toLowerCase().includes(searchQuery.toLowerCase()) ||
          r.reference_image_path.toLowerCase().includes(searchQuery.toLowerCase());

        if (!matchesQuery) return false;

        if (statusFilter === "found") return r.found !== 0;
        if (statusFilter === "absent") return r.found === 0;
        return true;
      })
    : [];

  const hasGroundTruth =
    batchData &&
    batchData.results.some(
      (r) => r.gt_x !== undefined && r.gt_x !== null && r.loc_error !== undefined && r.loc_error !== null
    );

  const foundCount = batchData?.results.filter((r) => r.found !== 0).length || 0;
  const absentCount = batchData?.results.filter((r) => r.found === 0).length || 0;
  const detectedScales = batchData?.results.filter((r) => r.found !== 0 && r.scale).map((r) => r.scale as number) || [];
  const avgScale = detectedScales.length
    ? (detectedScales.reduce((a, b) => a + b, 0) / detectedScales.length).toFixed(1)
    : "—";

  return (
    <section className="border-t border-gray-100 bg-white py-10">
      <div className="mx-auto max-w-7xl px-8">
        {/* Header & Upload Controls */}
        <div className="mb-8 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h3 className="text-xl font-light tracking-tight text-[#1A1A2E]">
              Batch Registration & Pose Estimation
            </h3>
            <p className="mt-1 text-xs text-gray-500 max-w-2xl">
              Upload an unlabelled <span className="font-mono font-semibold text-[#1E3A5F]">pairs.csv</span> (containing <span className="font-mono text-gray-700">pair_id, search_path, reference_path</span>). DriftSense executes sub-pixel registration and estimates target pose parameters: <span className="font-mono text-[#1E3A5F]">x, y, theta, scale, found, score</span>.
            </p>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center gap-3">
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              className="hidden"
              onChange={handleFileChange}
            />

            <Button
              variant="outline"
              size="sm"
              onClick={() => fileInputRef.current?.click()}
              className="gap-2 text-xs font-medium border-gray-300"
            >
              <FileSpreadsheet className="h-4 w-4 text-emerald-600" />
              {file ? file.name : "Select pairs.csv"}
            </Button>

            <Button
              onClick={() => handleRunEvaluation()}
              disabled={!file || isLoading}
              className="gap-2 bg-[#1E3A5F] text-white hover:bg-[#152d4a] text-xs font-semibold h-9 px-5 shadow-sm"
            >
              <Upload className="h-3.5 w-3.5" />
              {isLoading ? "LOCALIZING..." : "RUN REGISTRATION"}
            </Button>
          </div>
        </div>

        {errorMsg && (
          <div className="mb-6 flex items-center gap-2 rounded border border-red-200 bg-red-50 p-3 text-xs text-red-700">
            <AlertCircle className="h-4 w-4 flex-shrink-0" />
            <span>{errorMsg}</span>
          </div>
        )}

        {/* Results Section */}
        {batchData && (
          <div className="space-y-8 animate-fade-in">
            {/* Top KPI Telemetry Cards */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <div className="rounded-lg border border-gray-200 bg-gray-50/60 p-4">
                <p className="text-[10px] font-semibold uppercase tracking-wider text-gray-400">
                  Total Image Pairs
                </p>
                <p className="mt-1 font-mono text-2xl font-bold text-[#1A1A2E]">
                  {batchData.summary.total_pairs}
                </p>
                <p className="mt-0.5 text-[11px] text-gray-400">Evaluated on disk</p>
              </div>

              <div className="rounded-lg border border-gray-200 bg-gray-50/60 p-4">
                <p className="text-[10px] font-semibold uppercase tracking-wider text-gray-400">
                  Target Found (Present)
                </p>
                <p className="mt-1 font-mono text-2xl font-bold text-emerald-600">
                  {foundCount} <span className="text-xs font-normal text-gray-400">/ {batchData.summary.total_pairs}</span>
                </p>
                <p className="mt-0.5 text-[11px] text-emerald-600 font-medium">
                  {((foundCount / batchData.summary.total_pairs) * 100).toFixed(1)}% detection rate
                </p>
              </div>

              <div className="rounded-lg border border-gray-200 bg-gray-50/60 p-4">
                <p className="text-[10px] font-semibold uppercase tracking-wider text-gray-400">
                  Absent / Decoys
                </p>
                <p className="mt-1 font-mono text-2xl font-bold text-gray-700">
                  {absentCount} <span className="text-xs font-normal text-gray-400">/ {batchData.summary.total_pairs}</span>
                </p>
                <p className="mt-0.5 text-[11px] text-gray-500 font-medium">
                  {((absentCount / batchData.summary.total_pairs) * 100).toFixed(1)}% rejected (τ=0.55)
                </p>
              </div>

              <div className="rounded-lg border border-gray-200 bg-gray-50/60 p-4">
                <p className="text-[10px] font-semibold uppercase tracking-wider text-gray-400">
                  Mean Scale Zoom
                </p>
                <p className="mt-1 font-mono text-2xl font-bold text-[#1A1A2E]">
                  {avgScale}<span className="text-xs font-normal text-gray-400">×</span>
                </p>
                <p className="mt-0.5 text-[11px] text-gray-400">Scale range: 8.0× - 12.0×</p>
              </div>

              <div className="rounded-lg border border-gray-200 bg-gray-50/60 p-4">
                <p className="text-[10px] font-semibold uppercase tracking-wider text-gray-400">
                  Avg Inference Speed
                </p>
                <p className="mt-1 font-mono text-2xl font-bold text-[#1E3A5F]">
                  {batchData.summary.avg_inference_time_s.toFixed(2)}s
                </p>
                <p className="mt-0.5 text-[11px] text-gray-400">
                  {(batchData.summary.avg_inference_time_s * 1000).toFixed(0)} ms / pair
                </p>
              </div>
            </div>

            {/* If Ground-Truth is available in CSV, show validation metrics */}
            {hasGroundTruth && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-slate-50 p-5 rounded-lg border border-slate-200">
                <div>
                  <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">
                    Ground-Truth Validation Accuracy
                  </h4>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-[11px] text-gray-400">Median Localization Error</p>
                      <p className="font-mono text-xl font-bold text-emerald-600">
                        {batchData.summary.median_loc_error?.toFixed(4)} px
                      </p>
                    </div>
                    <div>
                      <p className="text-[11px] text-gray-400">Accuracy (≤ 1.0 px)</p>
                      <p className="font-mono text-xl font-bold text-emerald-600">
                        {batchData.summary.accuracy_breakdown[0]?.accuracy_pct.toFixed(1)}%
                      </p>
                    </div>
                  </div>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left font-mono text-xs">
                    <thead>
                      <tr className="border-b border-gray-200 text-gray-400">
                        <th className="pb-1">Tolerance</th>
                        <th className="pb-1">Correct</th>
                        <th className="pb-1">Failed</th>
                        <th className="pb-1">Accuracy</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200/50">
                      {batchData.summary.accuracy_breakdown.slice(0, 3).map((b) => (
                        <tr key={b.threshold_px}>
                          <td className="py-1">≤ {b.threshold_px.toFixed(1)} px</td>
                          <td className="py-1 text-emerald-600 font-semibold">{b.correct_count}</td>
                          <td className="py-1 text-gray-400">{b.failed_count}</td>
                          <td className="py-1 font-semibold">{b.accuracy_pct.toFixed(1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Individual Predictions Table Header with Filters and Download Button */}
            <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm">
              <div className="mb-4 flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-gray-100 pb-4">
                <div>
                  <h4 className="text-sm font-semibold uppercase tracking-wider text-[#1A1A2E] flex items-center gap-2">
                    <Crosshair className="h-4 w-4 text-[#1E3A5F]" />
                    Registration Predictions Output ({filteredResults.length} pairs)
                  </h4>
                  <p className="text-xs text-gray-400 mt-0.5">
                    Exact output format for <span className="font-mono text-gray-600">predictions.csv</span>: pair_id, x, y, theta, scale, found, score
                  </p>
                </div>

                <div className="flex items-center gap-3">
                  {/* Status filter */}
                  <div className="flex items-center rounded-md border border-gray-200 bg-gray-50 p-0.5 text-xs">
                    <button
                      onClick={() => setStatusFilter("all")}
                      className={`px-2.5 py-1 rounded font-medium transition-colors ${
                        statusFilter === "all" ? "bg-white text-[#1E3A5F] shadow-sm font-semibold" : "text-gray-500 hover:text-gray-900"
                      }`}
                    >
                      All ({batchData.results.length})
                    </button>
                    <button
                      onClick={() => setStatusFilter("found")}
                      className={`px-2.5 py-1 rounded font-medium transition-colors ${
                        statusFilter === "found" ? "bg-white text-emerald-700 shadow-sm font-semibold" : "text-gray-500 hover:text-gray-900"
                      }`}
                    >
                      Found ({foundCount})
                    </button>
                    <button
                      onClick={() => setStatusFilter("absent")}
                      className={`px-2.5 py-1 rounded font-medium transition-colors ${
                        statusFilter === "absent" ? "bg-white text-gray-700 shadow-sm font-semibold" : "text-gray-500 hover:text-gray-900"
                      }`}
                    >
                      Absent ({absentCount})
                    </button>
                  </div>

                  {/* Search query */}
                  <div className="relative w-48">
                    <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-gray-400" />
                    <Input
                      type="text"
                      placeholder="Filter pair_id..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="h-8 pl-8 text-xs font-mono"
                    />
                  </div>

                  {/* Download predictions.csv */}
                  <Button
                    onClick={handleDownloadPredictions}
                    size="sm"
                    className="gap-2 bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold h-8 px-4"
                  >
                    <Download className="h-3.5 w-3.5" />
                    Export predictions.csv
                  </Button>
                </div>
              </div>

              {/* Predictions Table */}
              <div className="max-h-[480px] overflow-y-auto rounded border border-gray-100">
                <table className="w-full text-left font-mono text-xs">
                  <thead className="sticky top-0 bg-gray-50 text-gray-500 border-b border-gray-200">
                    <tr>
                      <th className="p-2.5 font-semibold">pair_id</th>
                      <th className="p-2.5 font-semibold text-center">found</th>
                      <th className="p-2.5 font-semibold">x (px)</th>
                      <th className="p-2.5 font-semibold">y (px)</th>
                      <th className="p-2.5 font-semibold">theta (θ)</th>
                      <th className="p-2.5 font-semibold">scale (z)</th>
                      <th className="p-2.5 font-semibold">score (NCC)</th>
                      {hasGroundTruth && <th className="p-2.5 font-semibold">Loc Error</th>}
                      <th className="p-2.5 font-semibold">Search Path</th>
                      <th className="p-2.5 font-semibold">Reference Path</th>
                      <th className="p-2.5 font-semibold text-right">Time</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100 text-gray-700">
                    {filteredResults.map((r) => (
                      <tr key={r.index} className="hover:bg-gray-50/70 transition-colors">
                        <td className="p-2.5 font-bold text-[#1E3A5F]">
                          {r.pair_id || `pair_${r.index.toString().padStart(3, "0")}`}
                        </td>

                        <td className="p-2.5 text-center">
                          {r.found !== 0 ? (
                            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-50 text-emerald-700 border border-emerald-200">
                              <span className="h-1.5 w-1.5 rounded-full bg-emerald-600"></span>
                              1 (FOUND)
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold bg-gray-100 text-gray-600 border border-gray-200">
                              <span className="h-1.5 w-1.5 rounded-full bg-gray-400"></span>
                              0 (ABSENT)
                            </span>
                          )}
                        </td>

                        <td className="p-2.5 font-medium">
                          {r.found !== 0 && r.detected_x !== null && r.detected_x !== undefined ? (
                            <span className="text-gray-900 font-semibold">{r.detected_x.toFixed(1)}</span>
                          ) : (
                            <span className="text-gray-400">0.0</span>
                          )}
                        </td>

                        <td className="p-2.5 font-medium">
                          {r.found !== 0 && r.detected_y !== null && r.detected_y !== undefined ? (
                            <span className="text-gray-900 font-semibold">{r.detected_y.toFixed(1)}</span>
                          ) : (
                            <span className="text-gray-400">0.0</span>
                          )}
                        </td>

                        <td className="p-2.5">
                          {r.found !== 0 && r.rotation !== null && r.rotation !== undefined ? (
                            <span className="text-gray-800">
                              {r.rotation >= 0 ? "+" : ""}
                              {r.rotation.toFixed(1)}°
                            </span>
                          ) : (
                            <span className="text-gray-400">0.0°</span>
                          )}
                        </td>

                        <td className="p-2.5">
                          {r.found !== 0 && r.scale !== null && r.scale !== undefined ? (
                            <span className="text-gray-800">{r.scale.toFixed(1)}×</span>
                          ) : (
                            <span className="text-gray-400">0.0×</span>
                          )}
                        </td>

                        <td className="p-2.5">
                          <span
                            className={`font-semibold ${
                              r.confidence >= 0.7
                                ? "text-emerald-700"
                                : r.confidence >= 0.55
                                ? "text-blue-700"
                                : "text-gray-500"
                            }`}
                          >
                            {r.confidence.toFixed(4)}
                          </span>
                        </td>

                        {hasGroundTruth && (
                          <td className="p-2.5">
                            {r.found === 0 ? (
                              <span className="text-gray-400 font-normal">N/A</span>
                            ) : r.loc_error !== undefined && r.loc_error !== null ? (
                              <span
                                className={`font-semibold ${
                                  r.loc_error <= 1.0
                                    ? "text-emerald-600"
                                    : r.loc_error <= 5.0
                                    ? "text-amber-600"
                                    : "text-red-600"
                                }`}
                              >
                                {r.loc_error.toFixed(2)} px
                              </span>
                            ) : (
                              "N/A"
                            )}
                          </td>
                        )}

                        <td className="p-2.5 truncate max-w-[140px] text-gray-500" title={r.search_image_path}>
                          {r.search_image_path}
                        </td>

                        <td className="p-2.5 truncate max-w-[140px] text-gray-500" title={r.reference_image_path}>
                          {r.reference_image_path}
                        </td>

                        <td className="p-2.5 text-right text-gray-400">
                          {r.elapsed_s.toFixed(2)}s
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
