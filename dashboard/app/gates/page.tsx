"use client";

import { motion } from "framer-motion";
import { useState } from "react";
import { mutate } from "swr";

import { EmptyState, Panel, SectionTitle } from "@/components/primitives";
import { resolveGate, useGates } from "@/lib/api";
import { STAGE_BY_KEY } from "@/lib/stages";
import type { PendingGate } from "@/lib/types";

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

function GateSummary({ summary }: { summary: Record<string, unknown> }) {
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
      </div>
    );
  }

  return (
    <div className="grid gap-2 sm:grid-cols-2">
      {Object.entries(summary).map(([k, v]) => (
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
  );
}
