"use client";

import { motion } from "framer-motion";
import { useState } from "react";
import { mutate } from "swr";

import { EmptyState, Panel, SectionTitle } from "@/components/primitives";
import { resolveGate, useGates } from "@/lib/api";
import { STAGE_BY_KEY } from "@/lib/stages";
import type { DecisionRecord, DeploymentDossier, ExcludedTemplate, PendingGate } from "@/lib/types";

const GATE_TITLE: Record<string, { title: string; blurb: string }> = {
  gate_spec: {
    title: "Confirm the problem specification",
    blurb: "The Requirement agent interpreted your goal. Confirm the task, target, and metric before any data is touched.",
  },
  gate_plan: {
    title: "Approve the training plan & cost",
    blurb: "Baselines first, then candidates. Review the experiments and the estimated budget before training starts.",
  },
  gate_deploy: {
    title: "Sign off on deployment",
    blurb: "The bundle packages the model, its inference schema, a FastAPI app, and a drift baseline. Approve to ship.",
  },
};

export default function GatesPage() {
  const { data: gates } = useGates();
  const list = gates ?? [];

  return (
    <div>
      <SectionTitle sub="The three human checkpoints. Nothing proceeds past a gate until you decide.">
        Approval gates
      </SectionTitle>

      {list.length === 0 ? (
        <Panel>
          <EmptyState
            icon="⛴"
            title="No gates awaiting approval"
            hint="When a run reaches a human checkpoint — problem spec, training plan, or deployment — it appears here."
          />
        </Panel>
      ) : (
        <div className="space-y-6">
          {list.map((gate) => (
            <GateCard key={`${gate.ui_run_id}-${gate.stage}`} gate={gate} />
          ))}
        </div>
      )}
    </div>
  );
}

function GateCard({ gate }: { gate: PendingGate }) {
  const meta = STAGE_BY_KEY[gate.stage];
  const info = GATE_TITLE[gate.stage] ?? { title: gate.stage, blurb: "" };
  const [busy, setBusy] = useState<"approve" | "reject" | null>(null);
  const [notes, setNotes] = useState("");

  const act = async (approved: boolean) => {
    setBusy(approved ? "approve" : "reject");
    try {
      await resolveGate(gate.ui_run_id, approved, notes);
      await mutate("/api/gates");
    } finally {
      setBusy(null);
    }
  };

  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
      <Panel className="border-warn/30">
        <div className="flex items-start gap-4">
          <div className="grid h-12 w-12 flex-none place-items-center rounded-xl bg-warn/15 text-2xl text-warn">
            {meta?.icon ?? "⛴"}
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-3">
              <h3 className="text-lg font-bold text-ink">{info.title}</h3>
              <span className="mono text-ink-faint">{gate.ui_run_id}</span>
            </div>
            <p className="mt-1 text-sm text-ink-muted">{info.blurb}</p>
          </div>
        </div>

        <div className="mt-4 rounded-xl border border-base-600/50 bg-base-900/50 p-4">
          <div className="label-eyebrow mb-2">For your review</div>
          <GateSummary summary={gate.summary} />
        </div>

        <textarea
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="Optional notes (shown to the team on rejection)…"
          className="mt-4 min-h-16 w-full resize-none rounded-xl border border-base-600/70 bg-base-900/60 p-3 text-sm text-ink outline-none focus:border-accent/60"
        />

        <div className="mt-4 flex gap-3">
          <button onClick={() => act(true)} disabled={busy !== null} className="btn-ok flex-1 py-3">
            {busy === "approve" ? "Approving…" : "✓  Approve & continue"}
          </button>
          <button onClick={() => act(false)} disabled={busy !== null} className="btn-danger flex-1 py-3">
            {busy === "reject" ? "Rejecting…" : "✕  Reject"}
          </button>
        </div>
      </Panel>
    </motion.div>
  );
}

// Fields rendered by a dedicated block below, so the generic fallback grid
// doesn't also dump them as raw JSON.
const RENDERED_ELSEWHERE = new Set(["reasoning", "excluded_templates", "dossier"]);

function GateSummary({ summary }: { summary: Record<string, unknown> }) {
  const reasoning = isDecisionRecord(summary.reasoning) ? summary.reasoning : null;
  const excludedTemplates = Array.isArray(summary.excluded_templates)
    ? (summary.excluded_templates as ExcludedTemplate[])
    : [];
  const dossier = isDossier(summary.dossier) ? summary.dossier : null;

  // Render the most useful shapes nicely; fall back to JSON.
  if (Array.isArray(summary.experiments)) {
    const exps = summary.experiments as Record<string, unknown>[];
    return (
      <div>
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-ink-faint">
              <th className="pb-2">Experiment</th>
              <th className="pb-2">Template</th>
              <th className="pb-2">Est. seconds</th>
              <th className="pb-2">Baseline</th>
            </tr>
          </thead>
          <tbody>
            {exps.map((e, i) => (
              <tr key={i} className="border-t border-base-700/50">
                <td className="py-1.5 font-mono text-ink">{String(e.experiment_id)}</td>
                <td className="py-1.5 text-ink-muted">{String(e.template_id)}</td>
                <td className="py-1.5 text-ink-muted">{String(e.estimated_seconds ?? "—")}</td>
                <td className="py-1.5">{e.is_baseline ? "✓" : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {"estimated_total_seconds" in summary && (
          <div className="mt-3 flex gap-6 text-sm">
            <span className="text-ink-muted">
              Est. total:{" "}
              <span className="font-mono text-ink">
                {String(summary.estimated_total_seconds)}s
              </span>
            </span>
            {"llm_cost_so_far_usd" in summary && (
              <span className="text-ink-muted">
                LLM so far:{" "}
                <span className="font-mono text-ink">
                  ${Number(summary.llm_cost_so_far_usd).toFixed(4)}
                </span>
              </span>
            )}
          </div>
        )}
        {excludedTemplates.length > 0 && (
          <div className="mt-4">
            <div className="label-eyebrow mb-2">Deliberately not scheduled</div>
            <div className="space-y-1.5">
              {excludedTemplates.map((e) => (
                <div key={e.template_id} className="flex gap-2 text-sm">
                  <span className="chip">{e.template_id}</span>
                  <span className="text-ink-muted">{e.reason}</span>
                </div>
              ))}
            </div>
          </div>
        )}
        {reasoning && <ReasoningBlock reasoning={reasoning} className="mt-4" />}
      </div>
    );
  }

  return (
    <div>
      <div className="grid gap-2 sm:grid-cols-2">
        {Object.entries(summary)
          .filter(([k]) => !RENDERED_ELSEWHERE.has(k))
          .map(([k, v]) => (
            <div key={k} className="flex flex-col gap-0.5 rounded-lg bg-base-800/50 p-2.5">
              <span className="text-[11px] uppercase tracking-wide text-ink-faint">
                {k.replace(/_/g, " ")}
              </span>
              <span className="break-words font-mono text-xs text-ink">
                {typeof v === "object" ? JSON.stringify(v) : String(v)}
              </span>
            </div>
          ))}
      </div>
      {reasoning && <ReasoningBlock reasoning={reasoning} className="mt-4" />}
      {dossier && <DossierBlock dossier={dossier} className="mt-4" />}
    </div>
  );
}

function isDecisionRecord(value: unknown): value is DecisionRecord {
  return typeof value === "object" && value !== null && "decision" in value;
}

function isDossier(value: unknown): value is DeploymentDossier {
  return typeof value === "object" && value !== null && "comparison" in value;
}

const STATE_TEXT_CLASS: Record<string, string> = {
  completed: "text-ok",
  failed: "text-danger",
  skipped: "text-ink-faint",
};

function DossierBlock({
  dossier,
  className = "",
}: {
  dossier: DeploymentDossier;
  className?: string;
}) {
  return (
    <div className={className}>
      <div className="label-eyebrow mb-2">Why trust this model</div>

      {dossier.comparison.length > 0 && (
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-ink-faint">
              <th className="pb-2">Experiment</th>
              <th className="pb-2">Template</th>
              <th className="pb-2">State</th>
              <th className="pb-2">{dossier.comparison[0].primary_metric}</th>
            </tr>
          </thead>
          <tbody>
            {dossier.comparison.map((row) => (
              <tr
                key={row.experiment_id}
                className={`border-t border-base-700/50 ${row.is_winner ? "bg-accent/8" : ""}`}
              >
                <td className="py-1.5 font-mono text-ink">
                  {row.is_winner && (
                    <span className="mr-2 rounded bg-accent-grad px-1.5 py-0.5 text-[10px] font-bold text-white">
                      WINNER
                    </span>
                  )}
                  {row.experiment_id}
                </td>
                <td className="py-1.5 text-ink-muted">{row.template_id}</td>
                <td className={`py-1.5 ${STATE_TEXT_CLASS[row.state] ?? "text-ink-muted"}`}>
                  {row.state}
                </td>
                <td className="py-1.5 font-mono text-ink">
                  {row.mean != null ? `${row.mean.toFixed(4)} ± ${(row.std ?? 0).toFixed(3)}` : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {dossier.winner_margin && (
        <div className="mt-3 flex flex-wrap gap-6 text-sm">
          {dossier.winner_margin.vs_runner_up && (
            <span className="text-ink-muted">
              Lead over runner-up ({dossier.winner_margin.vs_runner_up.template_id}):{" "}
              <span className="font-mono text-ink">
                +{dossier.winner_margin.vs_runner_up.improvement.toFixed(4)}
              </span>
            </span>
          )}
          {dossier.winner_margin.vs_dummy != null && (
            <span className="text-ink-muted">
              Lead over dummy baseline:{" "}
              <span className="font-mono text-ink">
                +{dossier.winner_margin.vs_dummy.toFixed(4)}
              </span>
            </span>
          )}
        </div>
      )}

      {dossier.top_features.length > 0 && (
        <div className="mt-4">
          <div className="label-eyebrow mb-2">Top features</div>
          <div className="flex flex-wrap gap-2">
            {dossier.top_features.map((f) => (
              <span key={f.feature} className="chip">
                {f.feature} <span className="text-ink-faint">{f.importance.toFixed(3)}</span>
              </span>
            ))}
          </div>
        </div>
      )}

      {dossier.critique && (
        <div className="mt-4">
          <div className="label-eyebrow mb-2">
            Critique verdict:{" "}
            <span className="text-ink">{dossier.critique.verdict.replace(/_/g, " ")}</span>
          </div>
          {dossier.critique.findings.length > 0 && (
            <ul className="list-inside list-disc space-y-0.5 text-sm text-ink-muted">
              {dossier.critique.findings.map((f, i) => (
                <li key={i}>
                  <span className="uppercase text-ink-faint">{f.severity}</span> [{f.category}]{" "}
                  {f.evidence}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

function ReasoningBlock({
  reasoning,
  className = "",
}: {
  reasoning: DecisionRecord;
  className?: string;
}) {
  return (
    <div className={`rounded-lg border border-accent/20 bg-accent/5 p-3 ${className}`}>
      <div className="label-eyebrow mb-2 text-accent-bright">Why</div>
      {reasoning.why && <p className="text-sm text-ink">{reasoning.why}</p>}
      <ReasoningList label="Evidence" items={reasoning.evidence} />
      <ReasoningList label="Alternatives considered" items={reasoning.alternatives_considered} />
      <ReasoningList label="Assumptions" items={reasoning.assumptions} />
    </div>
  );
}

function ReasoningList({ label, items }: { label: string; items: string[] }) {
  if (!items || items.length === 0) return null;
  return (
    <div className="mt-2">
      <div className="text-[11px] uppercase tracking-wide text-ink-faint">{label}</div>
      <ul className="mt-1 list-inside list-disc space-y-0.5 text-sm text-ink-muted">
        {items.map((item, i) => (
          <li key={i}>{item}</li>
        ))}
      </ul>
    </div>
  );
}
