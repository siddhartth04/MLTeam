"use client";

import { motion } from "framer-motion";
import Link from "next/link";

import { ActivityFeed } from "@/components/ActivityFeed";
import { AgentFlow } from "@/components/AgentFlow";
import { useActiveRun } from "@/components/RunContext";
import { EmptyState, Panel, SectionTitle, StatTile, StatusPill } from "@/components/primitives";
import { useCritique, useRun } from "@/lib/api";
import { STAGE_BY_KEY } from "@/lib/stages";
import { useRunStream } from "@/lib/useRunStream";

export default function MissionPage() {
  const { runId } = useActiveRun();
  const stream = useRunStream(runId);
  const { data: run } = useRun(runId, 3000);
  const { data: critique } = useCritique(run?.status === "completed" ? runId : null);

  if (!runId) {
    return (
      <Panel>
        <EmptyState
          icon="◉"
          title="No run selected"
          hint="Launch a run or pick one from the sidebar to watch the agents work in real time."
        />
      </Panel>
    );
  }

  // Prefer live stream stages; fall back to the polled run snapshot.
  const stages = stream.stages.length ? stream.stages : (run?.stages ?? []);
  const done = stages.filter((s) => s.status === "done").length;
  const total = stages.length || 1;
  const pct = Math.round((done / total) * 100);
  const status = stream.status ?? run?.status ?? "queued";
  const cost = stream.cost || run?.llm_cost_usd || 0;
  const activeStage = stages.find((s) => s.status === "active" || s.status === "awaiting_approval");

  return (
    <div>
      <SectionTitle sub="A live X-ray of the autonomous team. Nodes pulse as agents work; the store is the hub every handoff flows through.">
        Mission Control
      </SectionTitle>

      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <StatTile
          label="Pipeline progress"
          value={`${pct}%`}
          sub={`${done} / ${total} stages`}
          accent="cyan"
        />
        <StatTile
          label="Now running"
          value={
            <span className="text-lg">
              {activeStage ? STAGE_BY_KEY[activeStage.stage]?.short ?? activeStage.label : "—"}
            </span>
          }
          sub={activeStage?.detail || "idle"}
          accent="accent"
        />
        <StatTile
          label="LLM spend"
          value={`$${cost.toFixed(4)}`}
          sub="running total"
          accent="ok"
        />
        <StatTile
          label="Status"
          value={<StatusPill status={status} />}
          sub={stream.connected ? "live stream connected" : "polling"}
          accent={status === "failed" || status === "rejected" ? "danger" : "warn"}
        />
      </div>

      {stream.gate && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass mt-4 flex items-center justify-between gap-4 border-warn/40 bg-warn/8 p-4"
        >
          <div className="flex items-center gap-3">
            <span className="grid h-9 w-9 place-items-center rounded-lg bg-warn/20 text-warn">⛴</span>
            <div>
              <div className="font-semibold text-ink">Human approval required</div>
              <div className="text-sm text-ink-muted">
                {STAGE_BY_KEY[stream.gate]?.label ?? stream.gate} is waiting for your sign-off.
              </div>
            </div>
          </div>
          <Link href={`/gates?run=${runId}`} className="btn-primary">
            Review &amp; decide →
          </Link>
        </motion.div>
      )}

      <div className="mt-6 grid grid-cols-1 gap-6 xl:grid-cols-[1.15fr_0.85fr]">
        <Panel eyebrow="Hub-and-spoke" title="Agent flow">
          <AgentFlow stages={stages} />
          <div className="mt-3 flex flex-wrap justify-center gap-4 text-xs text-ink-faint">
            <Legend color="#7c5cff" label="active" />
            <Legend color="#38e8a0" label="done" />
            <Legend color="#ffb648" label="gate / awaiting" />
            <Legend color="#4a557a" label="skipped" />
            <Legend color="#2b3352" label="pending" />
          </div>
        </Panel>

        <div className="space-y-6">
          <Panel eyebrow="Live" title="Activity feed" className="pb-4">
            <ActivityFeed events={stream.events.length ? stream.events : []} height={340} />
          </Panel>
          {critique && (
            <Panel eyebrow="Outcome" title="Critic verdict">
              <div className="flex items-center gap-3">
                <VerdictInline verdict={critique.verdict} />
              </div>
              {critique.iteration_actions.length > 0 && (
                <ul className="mt-3 space-y-1.5 text-sm text-ink-muted">
                  {critique.iteration_actions.map((a, i) => (
                    <li key={i} className="flex gap-2">
                      <span className="text-accent-bright">→</span>
                      {a}
                    </li>
                  ))}
                </ul>
              )}
            </Panel>
          )}
        </div>
      </div>

      {/* Stage rail */}
      <Panel eyebrow="Timeline" title="Every stage" className="mt-6">
        <div className="flex gap-2 overflow-x-auto pb-2">
          {stages.map((s) => {
            const meta = STAGE_BY_KEY[s.stage];
            const c =
              s.status === "done"
                ? "border-ok/50 text-ok"
                : s.status === "active"
                  ? "border-accent/60 text-accent-bright"
                  : s.status === "awaiting_approval"
                    ? "border-warn/60 text-warn"
                    : s.status === "skipped"
                      ? "border-base-500/50 text-ink-faint"
                      : s.status === "failed" || s.status === "rejected"
                        ? "border-danger/50 text-danger"
                        : "border-base-600/60 text-ink-faint";
            return (
              <div
                key={s.stage}
                className={`flex min-w-[120px] flex-none flex-col gap-1 rounded-xl border bg-base-800/50 px-3 py-2.5 ${c}`}
              >
                <div className="flex items-center gap-1.5 text-sm font-semibold">
                  <span>{meta?.icon}</span>
                  {meta?.short ?? s.label}
                </div>
                <div className="truncate text-[11px] text-ink-faint">
                  {s.detail || s.status.replace(/_/g, " ")}
                </div>
              </div>
            );
          })}
        </div>
      </Panel>
    </div>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="h-2.5 w-2.5 rounded-full" style={{ background: color }} />
      {label}
    </span>
  );
}

function VerdictInline({ verdict }: { verdict: string }) {
  const style: Record<string, string> = {
    approve: "bg-ok/15 text-ok border-ok/40",
    iterate_with_actions: "bg-warn/15 text-warn border-warn/40",
    halt: "bg-danger/15 text-danger border-danger/40",
  };
  return (
    <span
      className={`inline-flex items-center rounded-lg border px-3 py-1 text-sm font-bold ${style[verdict] ?? ""}`}
    >
      {verdict.replace(/_/g, " ")}
    </span>
  );
}
