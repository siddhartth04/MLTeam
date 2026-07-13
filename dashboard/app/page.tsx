"use client";

import { motion } from "framer-motion";
import { useRouter } from "next/navigation";
import { useCallback, useRef, useState } from "react";

import { EmptyState, Panel, SectionTitle } from "@/components/primitives";
import { AgentFlow } from "@/components/AgentFlow";
import { useActiveRun } from "@/components/RunContext";
import { launchRun } from "@/lib/api";

const EXAMPLES = [
  "Identify customers likely to churn so retention can call them first.",
  "Predict which loan applicants will default, minimizing costly false negatives.",
  "Flag transactions that are probably fraudulent for manual review.",
  "Estimate house sale prices from listing features.",
];

export default function LaunchPage() {
  const router = useRouter();
  const { setRunId } = useActiveRun();
  const [goal, setGoal] = useState("");
  const [runName, setRunName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [drag, setDrag] = useState(false);
  const [demo, setDemo] = useState(true);
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Demo mode replays a FIXED scenario scripted for the built-in churn
  // dataset, regardless of what's uploaded — combining it with a real file
  // produces a confusing failure several steps downstream (the requirement
  // agent proposes a target column, e.g. "churned", that doesn't exist in
  // the real data). Attaching a file always forces Live Agents so this
  // mismatch can't happen from the dashboard.
  const setFileAndForceLive = useCallback((f: File | null) => {
    setFile(f);
    if (f) setDemo(false);
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDrag(false);
      const f = e.dataTransfer.files?.[0];
      if (f) setFileAndForceLive(f);
    },
    [setFileAndForceLive],
  );

  const launch = async () => {
    if (!goal.trim()) {
      setError("Describe your goal to launch a run.");
      return;
    }
    setLaunching(true);
    setError(null);
    try {
      const id = await launchRun({ goalText: goal.trim(), file, demo, name: runName.trim() });
      setRunId(id);
      router.push(`/mission?run=${id}`);
    } catch (e) {
      setError(String(e));
      setLaunching(false);
    }
  };

  return (
    <div>
      <SectionTitle sub="Upload a dataset, describe the outcome you want, and hand it to the team.">
        Launch a run
      </SectionTitle>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-[1.1fr_0.9fr]">
        <div className="space-y-6">
          <Panel eyebrow="Step 1" title="Your goal">
            <input
              type="text"
              value={runName}
              onChange={(e) => setRunName(e.target.value)}
              placeholder="Run name (optional) — e.g. Churn v2"
              className="mb-3 w-full rounded-xl border border-base-600/70 bg-base-900/60 p-3 text-sm text-ink outline-none transition-colors placeholder:text-ink-faint focus:border-accent/60"
            />
            <textarea
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              placeholder="e.g. Identify customers likely to churn so retention can reach them first…"
              className="min-h-28 w-full resize-none rounded-xl border border-base-600/70 bg-base-900/60 p-4 text-ink outline-none transition-colors placeholder:text-ink-faint focus:border-accent/60"
            />
            <div className="mt-3 flex flex-wrap gap-2">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  onClick={() => setGoal(ex)}
                  className="chip transition-colors hover:border-accent/50 hover:text-ink"
                >
                  {ex.length > 46 ? ex.slice(0, 46) + "…" : ex}
                </button>
              ))}
            </div>
          </Panel>

          <Panel eyebrow="Step 2" title="Dataset">
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDrag(true);
              }}
              onDragLeave={() => setDrag(false)}
              onDrop={onDrop}
              onClick={() => inputRef.current?.click()}
              className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed p-8 text-center transition-all ${
                drag ? "border-accent bg-accent/8" : "border-base-500/60 bg-base-900/40 hover:border-accent/50"
              }`}
            >
              <input
                ref={inputRef}
                type="file"
                accept=".csv,.parquet"
                className="hidden"
                onChange={(e) => setFileAndForceLive(e.target.files?.[0] ?? null)}
              />
              <div className="text-3xl text-accent-bright/70">⬇</div>
              {file ? (
                <div className="text-ink">
                  <span className="font-semibold">{file.name}</span>
                  <span className="ml-2 text-xs text-ink-faint">
                    {(file.size / 1024).toFixed(0)} KB
                  </span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setFileAndForceLive(null);
                      if (inputRef.current) inputRef.current.value = "";
                    }}
                    className="ml-2 text-xs text-ink-faint underline hover:text-danger"
                  >
                    remove
                  </button>
                </div>
              ) : (
                <>
                  <div className="text-ink-muted">Drop a CSV or Parquet file, or click to browse</div>
                  <div className="text-xs text-ink-faint">
                    No file? We&apos;ll generate a synthetic churn dataset for you.
                  </div>
                </>
              )}
            </div>
          </Panel>

          <Panel eyebrow="Step 3" title="Execution mode">
            <div className="grid grid-cols-2 gap-3">
              <button
                onClick={() => !file && setDemo(true)}
                disabled={!!file}
                title={file ? "Demo replays a fixed scenario and can't process an uploaded dataset" : undefined}
                className={`rounded-xl border p-4 text-left transition-all ${
                  file
                    ? "cursor-not-allowed border-base-600 bg-base-700/20 opacity-50"
                    : demo
                      ? "border-accent/60 bg-accent/10"
                      : "border-base-500/60 bg-base-700/40"
                }`}
              >
                <div className="font-semibold text-ink">Demo (replay)</div>
                <div className="mt-1 text-xs text-ink-muted">
                  Recorded agent decisions — no API key, instant, deterministic.
                </div>
              </button>
              <button
                onClick={() => setDemo(false)}
                className={`rounded-xl border p-4 text-left transition-all ${
                  !demo ? "border-accent/60 bg-accent/10" : "border-base-500/60 bg-base-700/40"
                }`}
              >
                <div className="font-semibold text-ink">Live Agents</div>
                <div className="mt-1 text-xs text-ink-muted">
                  Real LLM agents. Falls back to replay if no API key is set.
                </div>
              </button>
            </div>
            {file && (
              <p className="mt-2 text-xs text-ink-faint">
                Demo replays a fixed scenario scripted for the built-in dataset, so it's disabled while a file is
                attached — remove the upload to use it.
              </p>
            )}

            {error && <div className="mt-4 text-sm text-danger">{error}</div>}

            <motion.button
              whileTap={{ scale: 0.98 }}
              onClick={launch}
              disabled={launching}
              className="btn-primary mt-5 w-full py-3 text-base"
            >
              {launching ? "Launching…" : "▶  Launch the team"}
            </motion.button>
            <p className="mt-3 text-center text-xs text-ink-faint">
              You&apos;ll approve the problem spec, the training plan &amp; cost, and the final
              deployment — three human gates.
            </p>
          </Panel>
        </div>

        <div className="space-y-6">
          <Panel eyebrow="The system" title="What happens next">
            <AgentFlow stages={[]} />
            <p className="mt-2 text-center text-xs text-ink-faint">
              Eleven specialized agents coordinate through a content-addressed artifact store.
              You&apos;ll watch them light up on Mission Control.
            </p>
          </Panel>
        </div>
      </div>
    </div>
  );
}
