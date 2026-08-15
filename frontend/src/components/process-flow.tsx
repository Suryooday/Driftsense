"use client";

const steps = [
  {
    num: "01",
    title: "Pattern Matching",
    desc: "Normalized cross-correlation searches for the reference structure.",
  },
  {
    num: "02",
    title: "Pose Refinement",
    desc: "Local optimization improves rotation and scale estimation.",
  },
  {
    num: "03",
    title: "Drift Detection",
    desc: "Expected and detected coordinates are compared in pixel space.",
  },
  {
    num: "04",
    title: "Stage Recovery",
    desc: "The required correction vector is calculated for repositioning.",
  },
];

export function ProcessFlow() {
  return (
    <section className="border-t border-gray-100 bg-gray-50/50 py-14">
      <div className="mx-auto max-w-7xl px-8">
        <div className="grid grid-cols-4 gap-10">
          {steps.map((s) => (
            <div key={s.num}>
              <p className="font-mono text-2xl font-light text-gray-200">
                {s.num}
              </p>
              <h4 className="mt-2 text-sm font-semibold text-[#1A1A2E]">
                {s.title}
              </h4>
              <p className="mt-1.5 text-xs leading-relaxed text-gray-400">
                {s.desc}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
