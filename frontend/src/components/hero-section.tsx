"use client";

import { ArrowRight } from "lucide-react";

const steps = ["REFERENCE", "LOCALIZE", "REFINE", "RECOVER"];

export function HeroSection() {
  return (
    <section className="bg-white py-16">
      <div className="mx-auto max-w-7xl px-8">
        <h2 className="text-4xl font-light tracking-tight text-[#1A1A2E]">
          Detect Drift.
          <br />
          <span className="font-semibold">Recover Position.</span>
        </h2>

        <p className="mt-4 max-w-2xl text-sm leading-relaxed text-gray-500">
          DriftSense localizes a reference pattern inside a zoomed-out wafer
          inspection image and calculates the coordinate correction required to
          recover the intended inspection site.
        </p>

        <div className="mt-10 flex items-center gap-2">
          {steps.map((step, i) => (
            <div key={step} className="flex items-center gap-2">
              <div className="flex items-center justify-center rounded border border-gray-200 bg-gray-50 px-5 py-2">
                <span className="text-[11px] font-semibold tracking-widest text-[#1E3A5F]">
                  {step}
                </span>
              </div>
              {i < steps.length - 1 && (
                <ArrowRight className="h-3.5 w-3.5 text-gray-300" />
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
