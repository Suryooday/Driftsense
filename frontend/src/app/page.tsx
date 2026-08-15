"use client";

import { useState, useEffect, useCallback } from "react";
import type { AnalysisResult } from "@/types/analysis";
import { healthCheck, runDemo, analyzeAlignment } from "@/lib/api";

import { Header } from "@/components/header";
import { HeroSection } from "@/components/hero-section";
import { ReferenceViewer } from "@/components/reference-viewer";
import { SearchViewer } from "@/components/search-viewer";
import { AnalysisControls } from "@/components/analysis-controls";
import { AlignmentResults } from "@/components/alignment-results";
import { DriftStatus } from "@/components/drift-status";
import { StageRecovery } from "@/components/stage-recovery";
import { ProcessFlow } from "@/components/process-flow";

export default function Home() {
  // State
  const [isOnline, setIsOnline] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const [refSrc, setRefSrc] = useState<string | null>(null);
  const [srchSrc, setSrchSrc] = useState<string | null>(null);
  const [refFile, setRefFile] = useState<File | null>(null);
  const [srchFile, setSrchFile] = useState<File | null>(null);

  const [expectedX, setExpectedX] = useState("");
  const [expectedY, setExpectedY] = useState("");

  const [result, setResult] = useState<AnalysisResult | null>(null);

  // Health check on mount
  useEffect(() => {
    healthCheck()
      .then(() => setIsOnline(true))
      .catch(() => setIsOnline(false));
  }, []);

  // Demo mode
  const handleRunDemo = useCallback(async () => {
    setIsLoading(true);
    setIsAnalyzing(true);
    setResult(null);

    try {
      const demo = await runDemo();

      // Set images
      setRefSrc(`data:image/png;base64,${demo.reference_image_b64}`);
      setSrchSrc(`data:image/png;base64,${demo.search_image_b64}`);
      setRefFile(null);
      setSrchFile(null);

      // Set expected coords
      setExpectedX(demo.analysis.expected.x.toFixed(2));
      setExpectedY(demo.analysis.expected.y.toFixed(2));

      // Simulate scanning delay for visual effect
      await new Promise((r) => setTimeout(r, 1200));

      setIsAnalyzing(false);
      setResult(demo.analysis);
    } catch (e) {
      console.error("Demo failed:", e);
      setIsAnalyzing(false);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Manual analysis
  const handleAnalyze = useCallback(async () => {
    if (!refFile || !srchFile) return;

    setIsLoading(true);
    setIsAnalyzing(true);
    setResult(null);

    try {
      const res = await analyzeAlignment(
        refFile,
        srchFile,
        parseFloat(expectedX) || 0,
        parseFloat(expectedY) || 0
      );

      await new Promise((r) => setTimeout(r, 800));
      setIsAnalyzing(false);
      setResult(res);
    } catch (e) {
      console.error("Analysis failed:", e);
      setIsAnalyzing(false);
    } finally {
      setIsLoading(false);
    }
  }, [refFile, srchFile, expectedX, expectedY]);

  // File uploads
  const handleRefUpload = (f: File) => {
    setRefFile(f);
    setRefSrc(URL.createObjectURL(f));
    setResult(null);
  };

  const handleSrchUpload = (f: File) => {
    setSrchFile(f);
    setSrchSrc(URL.createObjectURL(f));
    setResult(null);
  };

  return (
    <main className="min-h-screen bg-white">
      <Header
        isOnline={isOnline}
        onRunDemo={handleRunDemo}
        isLoading={isLoading}
      />

      <HeroSection />

      {/* Analysis Workspace */}
      <section className="border-t border-gray-100 bg-white py-10">
        <div className="mx-auto max-w-7xl px-8">
          <div className="grid grid-cols-2 gap-12">
            <ReferenceViewer imageSrc={refSrc} />
            <SearchViewer
              imageSrc={srchSrc}
              detected={result ? result.detected : null}
              isAnalyzing={isAnalyzing}
            />
          </div>
        </div>
      </section>

      {/* Controls */}
      <AnalysisControls
        expectedX={expectedX}
        expectedY={expectedY}
        onExpectedXChange={setExpectedX}
        onExpectedYChange={setExpectedY}
        onAnalyze={handleAnalyze}
        onReferenceUpload={handleRefUpload}
        onSearchUpload={handleSrchUpload}
        isLoading={isLoading}
        hasImages={!!(refFile && srchFile)}
      />

      {/* Results */}
      {result && (
        <>
          <AlignmentResults result={result} />
          <DriftStatus drift={result.drift} />
          <StageRecovery result={result} />
        </>
      )}

      <ProcessFlow />

      {/* Footer */}
      <footer className="border-t border-gray-100 bg-white py-6">
        <div className="mx-auto max-w-7xl px-8">
          <p className="text-[11px] text-gray-300">
            DriftSense · SEMICON India Hackathon 2026 · Classical NCC-based
            Matching + High-Resolution Pose Refinement
          </p>
        </div>
      </footer>
    </main>
  );
}
