"use client";

import { useState, useRef } from "react";
import type { CsvBatchResponse } from "@/types/analysis";
import { analyzeCsv } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Upload, FileSpreadsheet, CheckCircle2, AlertCircle, Search } from "lucide-react";
import { Input } from "@/components/ui/input";

export function CsvBatchViewer() {
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [batchData, setBatchData] = useState<CsvBatchResponse | null>(null);
  const [searchQuery, setSearchQuery] = useState("");

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) {
      setFile(selected);
      setErrorMsg(null);
    }
  };

  const handleRunEvaluation = async () => {
    if (!file) return;
    setIsLoading(true);
    setErrorMsg(null);

    try {
      const res = await analyzeCsv(file);
      setBatchData(res);
    } catch (err: any) {
      setErrorMsg(err.message || "Failed to analyze CSV dataset.");
    } finally {
      setIsLoading(false);
    }
  };

  const filteredResults = batchData
    ? batchData.results.filter(
        (r) =>
          r.search_image_path.toLowerCase().includes(searchQuery.toLowerCase()) ||
          r.reference_image_path.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : [];

  return (
    <section className="border-t border-gray-100 bg-white py-10">
      <div className="mx-auto max-w-7xl px-8">
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h3 className="text-xl font-light tracking-tight text-[#1A1A2E]">
              Batch Dataset CSV Evaluation
            </h3>
            <p className="mt-1 text-xs text-gray-400">
              Evaluate the frozen pattern matcher against an arbitrary ground truth CSV dataset file.
            </p>
          </div>

          {/* Upload & Run Controls */}
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
              className="gap-2 text-xs"
            >
              <FileSpreadsheet className="h-3.5 w-3.5 text-emerald-600" />
              {file ? file.name : "Select Test CSV"}
            </Button>

            <Button
              onClick={handleRunEvaluation}
              disabled={!file || isLoading}
              className="gap-2 bg-[#1E3A5F] text-white hover:bg-[#152d4a] text-xs font-semibold h-8 px-5"
            >
              <Upload className="h-3.5 w-3.5" />
              {isLoading ? "EVALUATING..." : "RUN EVALUATION"}
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
          <div className="space-y-10 animate-fade-in">
            {/* Top Summary Cards */}
            <div className="grid grid-cols-5 gap-6">
              <div className="rounded border border-gray-200 bg-gray-50/50 p-4">
                <p className="text-[10px] font-semibold uppercase tracking-widest text-gray-400">
                  Total Image Pairs
                </p>
                <p className="mt-1 font-mono text-2xl font-semibold text-[#1A1A2E]">
                  {batchData.summary.total_pairs}
                </p>
              </div>

              <div className="rounded border border-gray-200 bg-gray-50/50 p-4">
                <p className="text-[10px] font-semibold uppercase tracking-widest text-gray-400">
                  Accuracy (≤ 1.0 px)
                </p>
                <p className="mt-1 font-mono text-2xl font-semibold text-emerald-600">
                  {batchData.summary.accuracy_breakdown[0]?.accuracy_pct.toFixed(1)}%
                </p>
              </div>

              <div className="rounded border border-gray-200 bg-gray-50/50 p-4">
                <p className="text-[10px] font-semibold uppercase tracking-widest text-gray-400">
                  Mean Loc Error
                </p>
                <p className="mt-1 font-mono text-2xl font-semibold text-[#1A1A2E]">
                  {batchData.summary.mean_loc_error !== null && batchData.summary.mean_loc_error !== undefined
                    ? `${batchData.summary.mean_loc_error.toFixed(4)} px`
                    : "N/A"}
                </p>
              </div>

              <div className="rounded border border-gray-200 bg-gray-50/50 p-4">
                <p className="text-[10px] font-semibold uppercase tracking-widest text-gray-400">
                  Median Loc Error
                </p>
                <p className="mt-1 font-mono text-2xl font-semibold text-[#1A1A2E]">
                  {batchData.summary.median_loc_error !== null && batchData.summary.median_loc_error !== undefined
                    ? `${batchData.summary.median_loc_error.toFixed(4)} px`
                    : "N/A"}
                </p>
              </div>

              <div className="rounded border border-gray-200 bg-gray-50/50 p-4">
                <p className="text-[10px] font-semibold uppercase tracking-widest text-gray-400">
                  Avg Inference Speed
                </p>
                <p className="mt-1 font-mono text-2xl font-semibold text-[#1E3A5F]">
                  {(batchData.summary.avg_inference_time_s * 1000).toFixed(1)} ms
                </p>
              </div>
            </div>

            {/* Accuracy Breakdown Table & Confusion Matrix Grid */}
            <div className="grid grid-cols-2 gap-8">
              {/* Accuracy Breakdown Table */}
              <div className="rounded border border-gray-200 bg-white p-5">
                <h4 className="mb-3 text-xs font-semibold uppercase tracking-widest text-gray-400">
                  Accuracy Breakdown (1px - 5px Tolerance)
                </h4>
                <div className="overflow-x-auto">
                  <table className="w-full text-left font-mono text-xs">
                    <thead>
                      <tr className="border-b border-gray-100 text-gray-400">
                        <th className="pb-2 font-medium">Tolerance</th>
                        <th className="pb-2 font-medium">Correct</th>
                        <th className="pb-2 font-medium">Failed</th>
                        <th className="pb-2 font-medium">Accuracy</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-50 text-gray-700">
                      {batchData.summary.accuracy_breakdown.map((row) => (
                        <tr key={row.threshold_px}>
                          <td className="py-2">≤ {row.threshold_px.toFixed(1)} px</td>
                          <td className="py-2 text-emerald-600 font-semibold">{row.correct_count}</td>
                          <td className="py-2 text-gray-400">{row.failed_count}</td>
                          <td className="py-2 font-semibold">{row.accuracy_pct.toFixed(2)} %</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Confusion Matrix Buckets */}
              <div className="rounded border border-gray-200 bg-white p-5">
                <h4 className="mb-3 text-xs font-semibold uppercase tracking-widest text-gray-400">
                  Confusion Matrix / Error Bucket Distribution
                </h4>
                <div className="overflow-x-auto">
                  <table className="w-full text-left font-mono text-xs">
                    <thead>
                      <tr className="border-b border-gray-100 text-gray-400">
                        <th className="pb-2 font-medium">Bucket Range</th>
                        <th className="pb-2 font-medium">Count</th>
                        <th className="pb-2 font-medium">Percentage</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-50 text-gray-700">
                      {batchData.summary.confusion_matrix.map((row) => (
                        <tr key={row.bucket}>
                          <td className="py-2">{row.bucket}</td>
                          <td className="py-2 font-semibold">{row.count}</td>
                          <td className="py-2 text-gray-500">{row.percentage.toFixed(2)} %</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* Individual Pair Results Table */}
            <div className="rounded border border-gray-200 bg-white p-5">
              <div className="mb-4 flex items-center justify-between">
                <h4 className="text-xs font-semibold uppercase tracking-widest text-gray-400">
                  Individual Pair Localizations ({filteredResults.length} pairs)
                </h4>
                <div className="relative w-64">
                  <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-gray-400" />
                  <Input
                    type="text"
                    placeholder="Search image path..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="h-8 pl-8 text-xs font-mono"
                  />
                </div>
              </div>

              <div className="max-h-96 overflow-y-auto">
                <table className="w-full text-left font-mono text-xs">
                  <thead className="sticky top-0 bg-gray-50 text-gray-400 border-b border-gray-200">
                    <tr>
                      <th className="p-2 font-medium">Pair ID</th>
                      <th className="p-2 font-medium">Status</th>
                      <th className="p-2 font-medium">Search Image</th>
                      <th className="p-2 font-medium">Reference Image</th>
                      <th className="p-2 font-medium">Detected (X, Y)</th>
                      <th className="p-2 font-medium">GT (X, Y)</th>
                      <th className="p-2 font-medium">Loc Error</th>
                      <th className="p-2 font-medium">Rot (θ)</th>
                      <th className="p-2 font-medium">Scale (z)</th>
                      <th className="p-2 font-medium">Confidence</th>
                      <th className="p-2 font-medium">Latency</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100 text-gray-700">
                    {filteredResults.map((r) => (
                      <tr key={r.index} className="hover:bg-gray-50/50">
                        <td className="p-2 font-semibold text-gray-600">{r.pair_id || `#${r.index}`}</td>
                        <td className="p-2">
                          {r.found === 0 ? (
                            <span className="inline-block px-1.5 py-0.5 rounded text-[10px] font-semibold bg-gray-100 text-gray-700 border border-gray-200">
                              ABSENT / REJ
                            </span>
                          ) : r.loc_error !== undefined && r.loc_error !== null ? (
                            r.loc_error <= 1.0 ? (
                              <span className="inline-block px-1.5 py-0.5 rounded text-[10px] font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
                                ALIGNED
                              </span>
                            ) : r.loc_error <= 5.0 ? (
                              <span className="inline-block px-1.5 py-0.5 rounded text-[10px] font-semibold bg-amber-50 text-amber-700 border border-amber-200">
                                MINOR DRIFT
                              </span>
                            ) : (
                              <span className="inline-block px-1.5 py-0.5 rounded text-[10px] font-semibold bg-rose-50 text-rose-700 border border-rose-200">
                                LARGE DRIFT
                              </span>
                            )
                          ) : (
                            <span className="inline-block px-1.5 py-0.5 rounded text-[10px] font-semibold bg-blue-50 text-blue-700 border border-blue-200">
                              DETECTED
                            </span>
                          )}
                        </td>
                        <td className="p-2 truncate max-w-[120px]" title={r.search_image_path}>
                          {r.search_image_path.split("/").pop()}
                        </td>
                        <td className="p-2 truncate max-w-[120px]" title={r.reference_image_path}>
                          {r.reference_image_path.split("/").pop()}
                        </td>
                        <td className="p-2 font-semibold">
                          {r.found !== 0 && r.detected_x !== null && r.detected_y !== null
                            ? `(${r.detected_x?.toFixed(1)}, ${r.detected_y?.toFixed(1)})`
                            : "—"}
                        </td>
                        <td className="p-2 text-gray-500">
                          {r.gt_x !== undefined && r.gt_y !== undefined
                            ? `(${r.gt_x?.toFixed(1)}, ${r.gt_y?.toFixed(1)})`
                            : "N/A"}
                        </td>
                        <td className="p-2">
                          {r.found === 0 ? (
                            <span className="text-gray-400 font-normal">N/A (Absent)</span>
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
                        <td className="p-2 text-gray-600">
                          {r.rotation !== undefined && r.rotation !== null ? `${r.rotation >= 0 ? "+" : ""}${r.rotation.toFixed(1)}°` : "—"}
                        </td>
                        <td className="p-2 text-gray-600">
                          {r.scale !== undefined && r.scale !== null ? `${r.scale.toFixed(1)}×` : "—"}
                        </td>
                        <td className="p-2 font-mono text-gray-700">{r.confidence.toFixed(4)}</td>
                        <td className="p-2 text-gray-400">{(r.elapsed_s).toFixed(2)}s</td>
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
