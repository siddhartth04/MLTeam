"use client";

import { motion } from "framer-motion";

import { useActiveRun } from "@/components/RunContext";
import { EmptyState, Panel, SectionTitle } from "@/components/primitives";
import { useExperiments, useProblemSpec } from "@/lib/api";
import type { ExperimentRow } from "@/lib/types";

const HIGHER_BETTER = new Set(["roc_auc", "accuracy", "f1_macro", "r2"]);

export default function ExperimentsPage() {
  const { runId } = useActiveRun();
  const { data: exps } = useExperiments(runId);
  const { data: spec } = useProblemSpec(runId);

  if (!runId) {
    return (
      <Panel>
        <EmptyState icon="▤" title="No run selected" hint="Pick a run to see its experiment leaderboard." />
      </Panel>
    );
  }
  const rows = exps ?? [];
  const primary = spec?.primary_metric ?? "roc_auc";
  const metricNames = collectMetrics(rows, primary);

  // Best non-baseline by the primary metric.
  const scored = rows
    .filter((r) => r.metrics[primary])
    .map((r) => ({ r, v: r.metrics[primary].mean }));
  const bestId = pickBest(scored, primary)?.r.artifact_id;

  const maxByMetric = new Map<string, number>();
  for (const m of metricNames) {
    const vals = rows.map((r) => r.metrics[m]?.mean).filter((v): v is number => v != null);
    maxByMetric.set(m, Math.max(...vals, 1e-9));
  }

  return (
    <div>
      <SectionTitle sub="Every model the team trained — baselines, LightGBM variants, and any tuned or revised candidates.">
        Experiments
      </SectionTitle>

      {rows.length === 0 ? (
        <Panel>
          <EmptyState icon="▤" title="No experiments yet" hint="They appear once training runs." />
        </Panel>
      ) : (
        <Panel eyebrow="Leaderboard" title={`Ranked by ${primary}`}>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[720px] text-sm">
              <thead>
                <tr className="text-left text-ink-faint">
                  <th className="pb-3 pr-4">Experiment</th>
                  <th className="pb-3 pr-4">Template</th>
                  {metricNames.map((m) => (
                    <th key={m} className="pb-3 pr-4">
                      {m}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => {
                  const isBest = r.artifact_id === bestId;
                  return (
                    <motion.tr
                      key={r.artifact_id}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: i * 0.03 }}
                      className={`border-t border-base-700/50 ${
                        isBest ? "bg-accent/8" : ""
                      }`}
                    >
                      <td className="py-3 pr-4">
                        <div className="flex items-center gap-2">
                          {isBest && (
                            <span className="rounded bg-accent-grad px-1.5 py-0.5 text-[10px] font-bold text-white">
                              BEST
                            </span>
                          )}
                          <span className="font-mono text-ink">{r.experiment_id}</span>
                        </div>
                      </td>
                      <td className="py-3 pr-4">
                        <span className="chip">{r.template_id}</span>
                        {r.is_baseline && (
                          <span className="ml-2 text-[11px] text-ink-faint">baseline</span>
                        )}
                      </td>
                      {metricNames.map((m) => {
                        const cell = r.metrics[m];
                        if (!cell) return <td key={m} className="py-3 pr-4 text-ink-faint">—</td>;
                        const frac = HIGHER_BETTER.has(m)
                          ? cell.mean / (maxByMetric.get(m) || 1)
                          : (maxByMetric.get(m)! ? Math.min(1, 1 / (cell.mean / (maxByMetric.get(m) || 1))) : 0);
                        return (
                          <td key={m} className="py-3 pr-4">
                            <div className="flex items-center gap-2">
                              <span className="w-14 font-mono text-ink">{cell.mean.toFixed(4)}</span>
                              <div className="h-1.5 w-16 overflow-hidden rounded-full bg-base-700">
                                <div
                                  className="h-full rounded-full bg-accent-grad"
                                  style={{ width: `${Math.max(6, Math.min(100, frac * 100))}%` }}
                                />
                              </div>
                            </div>
                            <span className="text-[10px] text-ink-faint">±{cell.std.toFixed(3)}</span>
                          </td>
                        );
                      })}
                    </motion.tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>
      )}
    </div>
  );
}

function collectMetrics(rows: ExperimentRow[], primary: string): string[] {
  const set = new Set<string>();
  rows.forEach((r) => Object.keys(r.metrics).forEach((m) => set.add(m)));
  const all = Array.from(set);
  // primary first, then the rest sorted.
  return [primary, ...all.filter((m) => m !== primary).sort()].filter((m) => set.has(m));
}

function pickBest(scored: { r: ExperimentRow; v: number }[], primary: string) {
  if (scored.length === 0) return null;
  const higher = HIGHER_BETTER.has(primary);
  return scored.reduce((best, cur) =>
    (higher ? cur.v > best.v : cur.v < best.v) ? cur : best,
  );
}
