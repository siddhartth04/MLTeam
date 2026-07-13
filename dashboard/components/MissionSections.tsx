"use client";

// Concrete, animated Mission-Control sections: the pipeline as grouped
// phases, a spotlight on the agent working right now, a live leaderboard,
// and an explained artifact-store / lineage explorer. All driven by real
// API data (stages, experiments, lineage, artifact payloads).

import { AnimatePresence, motion } from "framer-motion";
import Link from "next/link";
import { useMemo, useState } from "react";

import { useArtifact, useExperiments, useLineage } from "@/lib/api";
import { STAGE_BY_KEY } from "@/lib/stages";
import type { StageState, StageStatus } from "@/lib/types";

import { EmptyState } from "./primitives";

// ---------------------------------------------------------------------------
// Pipeline, grouped into human phases.
// ---------------------------------------------------------------------------

interface Phase {
  eyebrow: string;
  name: string;
  keys: string[];
}

const PHASES: Phase[] = [
  { eyebrow: "Phase 1", name: "Understand the problem", keys: ["requirement", "gate_spec"] },
  { eyebrow: "Phase 2", name: "Prepare the data", keys: ["ingest", "profile", "split", "clean", "features"] },
  { eyebrow: "Phase 3", name: "Run experiments", keys: ["plan", "gate_plan", "train", "evaluate"] },
  { eyebrow: "Phase 4", name: "Judge & explain", keys: ["critic", "iterate", "explain"] },
  { eyebrow: "Phase 5", name: "Ship it", keys: ["deploy", "gate_deploy", "report"] },
];

const STAGE_CHIP: Record<StageStatus, { ring: string; text: string; dot: string }> = {
  done: { ring: "border-ok/40 bg-ok/8", text: "text-ok", dot: "bg-ok" },
  active: { ring: "border-accent/50 bg-accent/8", text: "text-accent-bright", dot: "bg-accent" },
  awaiting_approval: { ring: "border-warn/50 bg-warn/8", text: "text-warn", dot: "bg-warn" },
  failed: { ring: "border-danger/50 bg-danger/8", text: "text-danger", dot: "bg-danger" },
  rejected: { ring: "border-danger/50 bg-danger/8", text: "text-danger", dot: "bg-danger" },
  skipped: { ring: "border-base-500/50 bg-base-700/50", text: "text-ink-faint", dot: "bg-ink-faint" },
  pending: { ring: "border-base-600 bg-base-700/40", text: "text-ink-faint", dot: "bg-base-500" },
};

function useStatusOf(stages: StageState[]): (key: string) => StageStatus {
  return useMemo(() => {
    const m = new Map(stages.map((s) => [s.stage, s.status as StageStatus]));
    return (key: string) => m.get(key) ?? "pending";
  }, [stages]);
}

export function PhaseTimeline({ stages }: { stages: StageState[] }) {
  const statusOf = useStatusOf(stages);
  const detailOf = useMemo(() => {
    const m = new Map(stages.map((s) => [s.stage, s.detail]));
    return (key: string) => m.get(key) ?? "";
  }, [stages]);

  return (
    <div className="space-y-5">
      {PHASES.map((phase, pi) => {
        const statuses = phase.keys.map(statusOf);
        const doneCount = statuses.filter((s) => s === "done").length;
        const phaseActive = statuses.some((s) => s === "active" || s === "awaiting_approval");
        return (
          <motion.div
            key={phase.name}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: pi * 0.05 }}
          >
            <div className="mb-2 flex items-center gap-2.5">
              <span
                className={`grid h-6 w-6 flex-none place-items-center rounded-full text-[11px] font-bold ${
                  doneCount === phase.keys.length
                    ? "bg-ok text-white"
                    : phaseActive
                      ? "bg-accent text-white"
                      : "bg-base-700 text-ink-faint"
                }`}
              >
                {doneCount === phase.keys.length ? "✓" : pi + 1}
              </span>
              <span className="text-sm font-semibold text-ink">{phase.name}</span>
              <span className="mono text-[11px] text-ink-faint">
                {doneCount}/{phase.keys.length}
              </span>
              {phaseActive && (
                <span className="chip border-accent/40 bg-accent/8 text-accent-bright">running</span>
              )}
            </div>
            <div className="flex flex-wrap gap-2 pl-8">
              {phase.keys.map((key) => {
                const st = statusOf(key);
                const meta = STAGE_BY_KEY[key];
                const c = STAGE_CHIP[st];
                const detail = detailOf(key);
                return (
                  <div
                    key={key}
                    title={detail || st.replace(/_/g, " ")}
                    className={`flex items-center gap-2 rounded-xl border px-3 py-2 transition-colors ${c.ring}`}
                  >
                    <span className={`text-sm ${c.text}`}>{meta?.icon}</span>
                    <span className={`text-[13px] font-medium ${c.text}`}>{meta?.short ?? key}</span>
                    <span className="relative flex h-2 w-2">
                      {(st === "active" || st === "awaiting_approval") && (
                        <span className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-70 ${c.dot}`} />
                      )}
                      <span className={`relative inline-flex h-2 w-2 rounded-full ${c.dot}`} />
                    </span>
                  </div>
                );
              })}
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Spotlight on the agent working right now.
// ---------------------------------------------------------------------------

export function NowRunning({ active }: { active: StageState | undefined }) {
  if (!active) {
    return (
      <div className="flex items-center gap-4 rounded-2xl border border-base-600 bg-base-700/40 p-5">
        <span className="grid h-12 w-12 place-items-center rounded-xl bg-base-700 text-xl text-ink-faint">◌</span>
        <div>
          <div className="font-semibold text-ink">Idle</div>
          <div className="text-sm text-ink-muted">No agent is currently working on this run.</div>
        </div>
      </div>
    );
  }
  const meta = STAGE_BY_KEY[active.stage];
  const gate = active.status === "awaiting_approval";
  return (
    <motion.div
      key={active.stage}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className={`relative overflow-hidden rounded-2xl border p-5 ${
        gate ? "border-warn/40 bg-warn/8" : "border-accent/40 bg-accent/8"
      }`}
    >
      {/* sweeping shimmer to signal live work */}
      <motion.div
        aria-hidden
        className="pointer-events-none absolute inset-y-0 -left-1/3 w-1/3"
        style={{ background: "linear-gradient(90deg, transparent, rgba(99,102,241,0.14), transparent)" }}
        animate={{ x: ["0%", "400%"] }}
        transition={{ duration: 2.2, repeat: Infinity, ease: "linear" }}
      />
      <div className="relative flex items-center gap-4">
        <span
          className={`grid h-12 w-12 flex-none place-items-center rounded-xl text-xl text-white ${
            gate ? "bg-warn" : "bg-accent"
          }`}
        >
          {meta?.icon}
        </span>
        <div className="min-w-0">
          <div className="label-eyebrow">{gate ? "Waiting on you" : "Now running"}</div>
          <div className="truncate text-lg font-semibold text-ink">{meta?.label ?? active.label}</div>
          <div className="truncate text-sm text-ink-muted">{active.detail || (gate ? "Awaiting your approval" : "Working…")}</div>
        </div>
        {gate && (
          <span className="ml-auto flex-none animate-breathe text-2xl text-warn">⛴</span>
        )}
      </div>
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// Live leaderboard of trained models.
// ---------------------------------------------------------------------------

export function LeaderboardMini({ runId, metric }: { runId: string; metric: string | null }) {
  const { data: rows } = useExperiments(runId);
  const completed = (rows ?? []).filter((r) => r.state === "completed" && metric && r.metrics[metric]);
  if (!metric || completed.length === 0) {
    return <EmptyState icon="⚙" title="No trained models yet" hint="Experiments appear here as the team trains them." />;
  }
  const values = completed.map((r) => Math.abs(r.metrics[metric].mean));
  const max = Math.max(...values, 1e-9);
  const sorted = [...completed].sort((a, b) => Math.abs(b.metrics[metric].mean) - Math.abs(a.metrics[metric].mean));

  return (
    <div className="space-y-2.5">
      {sorted.map((r, i) => {
        const v = r.metrics[metric].mean;
        const pct = Math.max(4, (Math.abs(v) / max) * 100);
        return (
          <div key={r.artifact_id} className="flex items-center gap-3">
            <span className="w-40 flex-none truncate text-sm text-ink" title={r.template_id}>
              {r.template_id}
              {r.is_baseline && <span className="ml-1.5 rounded bg-base-700 px-1 text-[10px] text-ink-faint">base</span>}
            </span>
            <div className="relative h-6 flex-1 overflow-hidden rounded-md bg-base-700">
              <motion.div
                className="absolute inset-y-0 left-0 rounded-md bg-accent-grad"
                initial={{ width: 0 }}
                animate={{ width: `${pct}%` }}
                transition={{ delay: i * 0.05, type: "spring", stiffness: 120, damping: 20 }}
              />
            </div>
            <span className="w-20 flex-none text-right font-mono text-sm text-ink">{v.toFixed(4)}</span>
          </div>
        );
      })}
      <div className="pt-1 text-xs text-ink-faint">
        Bars are the <span className="font-semibold text-ink-muted">{metric}</span> of each trained model, relative to the best so far.
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Artifact store + lineage — explained, per run.
// ---------------------------------------------------------------------------

const ARTIFACT_INFO: Record<string, { blurb: string; color: string }> = {
  ProblemSpec: { blurb: "The confirmed problem definition — target, task type, metric, tolerances.", color: "#4f46e5" },
  DatasetVersion: { blurb: "A content-hashed snapshot of the data at one pipeline stage.", color: "#0891b2" },
  TriageReport: { blurb: "Data-readiness fixes applied before modeling, with affected-row counts.", color: "#0891b2" },
  DataProfile: { blurb: "Per-column stats, correlations, and target-leakage signals.", color: "#059669" },
  SplitPlan: { blurb: "How rows were split into train and a frozen holdout.", color: "#d97706" },
  CleaningManifest: { blurb: "The structural cleaning operations applied to the data.", color: "#7c3aed" },
  FeaturePlan: { blurb: "Row-local, leakage-safe feature-engineering operations.", color: "#0d9488" },
  FeatureSet: { blurb: "The engineered feature columns produced from the plan.", color: "#0d9488" },
  ExperimentPlan: { blurb: "Which model templates to train, in what order, at what budget.", color: "#c026d3" },
  ExperimentOutcome: { blurb: "Terminal state and CV metrics for one trained experiment.", color: "#0891b2" },
  TrainedModel: { blurb: "A fully serialized pipeline — preprocessing + model in one object.", color: "#059669" },
  EvaluationReport: { blurb: "CV / holdout metrics, baseline deltas, and leakage checks.", color: "#0891b2" },
  ExplainabilityReport: { blurb: "Permutation importances and effect figures.", color: "#ca8a04" },
  CritiqueReport: { blurb: "The Critic's verdict, findings, and any iteration actions.", color: "#dc2626" },
  InsightNarrative: { blurb: "Plain-language findings distilled for a human reader.", color: "#4f46e5" },
  DeploymentBundle: { blurb: "The generated serving app, Dockerfile, and drift baseline.", color: "#16a34a" },
  FinalReport: { blurb: "The business and technical HTML reports.", color: "#4f46e5" },
};

function infoFor(type: string) {
  return ARTIFACT_INFO[type] ?? { blurb: "A content-addressed artifact in the store.", color: "#586173" };
}

export function ArtifactStore({ runId }: { runId: string }) {
  const { data: graph } = useLineage(runId);
  const [selected, setSelected] = useState<string | null>(null);
  const { data: detail } = useArtifact(selected);

  const nodes = graph?.nodes ?? [];

  if (nodes.length === 0) {
    return (
      <EmptyState
        icon="❖"
        title="No artifacts yet"
        hint="Every decision the team makes is stored as a content-addressed artifact. They stream in here as the run produces them."
      />
    );
  }

  return (
    <div className="grid grid-cols-1 gap-5 lg:grid-cols-[1fr_1fr]">
      {/* left: the store, newest last, click to inspect */}
      <div>
        <div className="mb-3 flex items-center justify-between">
          <span className="text-sm text-ink-muted">
            <span className="font-semibold text-ink">{nodes.length}</span> artifacts produced
          </span>
          <Link href={`/lineage?run=${runId}`} className="text-xs font-medium text-accent-bright hover:underline">
            Full lineage graph →
          </Link>
        </div>
        <div className="max-h-[420px] space-y-1.5 overflow-y-auto pr-1">
          <AnimatePresence initial={false}>
            {nodes.map((n, i) => {
              const info = infoFor(n.type);
              const active = n.id === selected;
              return (
                <motion.button
                  key={n.id}
                  layout
                  initial={{ opacity: 0, x: -6 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: Math.min(i * 0.015, 0.3) }}
                  onClick={() => setSelected(n.id)}
                  className={`flex w-full items-center gap-3 rounded-xl border px-3 py-2 text-left transition-colors ${
                    active ? "border-accent/50 bg-accent/8" : "border-base-600 bg-base-800 hover:bg-base-700/60"
                  }`}
                >
                  <span className="h-8 w-1 flex-none rounded-full" style={{ background: info.color }} />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="truncate text-sm font-semibold text-ink">{n.type}</span>
                      {n.protected && <span title="Protected holdout">🔒</span>}
                    </div>
                    <div className="mono truncate text-[11px] text-ink-faint">{n.producer} · {n.short_id}</div>
                  </div>
                  <span className="flex-none text-ink-faint">›</span>
                </motion.button>
              );
            })}
          </AnimatePresence>
        </div>
      </div>

      {/* right: inspector — what it is + its real payload */}
      <div className="rounded-xl border border-base-600 bg-base-900/40 p-4">
        {!detail ? (
          <div className="grid h-full min-h-[200px] place-items-center text-center">
            <div>
              <div className="mb-2 text-3xl text-ink-faint opacity-60">◌</div>
              <div className="text-sm text-ink-muted">Click any artifact to inspect it.</div>
              <div className="mx-auto mt-1 max-w-xs text-xs text-ink-faint">
                The store is content-addressed: identical work produces an identical id, so every result is reproducible and traceable.
              </div>
            </div>
          </div>
        ) : (
          <motion.div key={detail.id} initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            <div className="flex items-center gap-2">
              <span className="h-4 w-1.5 rounded-full" style={{ background: infoFor(detail.type).color }} />
              <span className="text-base font-semibold text-ink">{detail.type}</span>
              {detail.protected && <span title="Protected holdout">🔒</span>}
            </div>
            <p className="mt-1 text-sm text-ink-muted">{infoFor(detail.type).blurb}</p>
            <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
              <div className="rounded-lg bg-base-800 p-2">
                <div className="label-eyebrow">Producer</div>
                <div className="mt-0.5 text-ink">{detail.producer}</div>
              </div>
              <div className="rounded-lg bg-base-800 p-2">
                <div className="label-eyebrow">Lineage</div>
                <div className="mt-0.5 text-ink">{detail.parents.length} in · {detail.children.length} out</div>
              </div>
            </div>
            <div className="mt-3">
              <div className="label-eyebrow mb-1">Payload</div>
              <pre className="max-h-64 overflow-auto rounded-lg border border-base-600 bg-base-800 p-3 text-[11px] leading-relaxed text-ink-muted">
                {JSON.stringify(detail.payload, null, 2)}
              </pre>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
