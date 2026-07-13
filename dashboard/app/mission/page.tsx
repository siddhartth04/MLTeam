"use client";

import { motion } from "framer-motion";
import Link from "next/link";

import { ActivityFeed } from "@/components/ActivityFeed";
import { AgentFlow } from "@/components/AgentFlow";
import { ArtifactStore, LeaderboardMini, NowRunning, PhaseTimeline } from "@/components/MissionSections";
import { useActiveRun } from "@/components/RunContext";
import { EmptyState, Panel, SectionTitle, StatTile, StatusPill, VerdictBadge } from "@/components/primitives";
import { useCritique, useProblemSpec, useRun } from "@/lib/api";
import { STAGE_BY_KEY } from "@/lib/stages";
import { useRunStream } from "@/lib/useRunStream";

export default function MissionPage() {
  const { runId } = useActiveRun();
  const stream = useRunStream(runId);
  const { data: run } = useRun(runId, 3000);
  const { data: spec } = useProblemSpec(runId);
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

  const stages = stream.stages.length ? stream.stages : (run?.stages ?? []);
  const done = stages.filter((s) => s.status === "done").length;
  const total = stages.length || 1;
  const pct = Math.round((done / total) * 100);
  const status = stream.status ?? run?.status ?? "queued";
  const cost = stream.cost || run?.llm_cost_usd || 0;
  const activeStage = stages.find((s) => s.status === "active" || s.status === "awaiting_approval");
  const metric = spec?.primary_metric ?? null;

  return (
    <div className="space-y-6">
      <SectionTitle sub="A live, sectioned view of the autonomous team — what each agent is doing right now, the models it trains, and every artifact it stores along the way.">
        Mission Control
      </SectionTitle>

      {/* ---- hero stats ---- */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <StatTile label="Pipeline progress" value={`${pct}%`} sub={`${done} / ${total} stages`} accent="cyan" />
        <StatTile
          label="Now running"
          value={<span className="text-lg">{activeStage ? STAGE_BY_KEY[activeStage.stage]?.short ?? activeStage.label : "—"}</span>}
          sub={activeStage?.detail || "idle"}
          accent="accent"
        />
        <StatTile label="LLM spend" value={`$${cost.toFixed(4)}`} sub="running total" accent="ok" />
        <StatTile
          label="Status"
          value={<StatusPill status={status} />}
          sub={stream.connected ? "live stream connected" : "polling"}
          accent={status === "failed" || status === "rejected" ? "danger" : "warn"}
        />
      </div>

      {/* ---- animated progress bar ---- */}
      <div className="h-2 w-full overflow-hidden rounded-full bg-base-700">
        <motion.div
          className="h-full rounded-full bg-accent-grad"
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ type: "spring", stiffness: 90, damping: 20 }}
        />
      </div>

      {/* ---- gate banner ---- */}
      {stream.gate && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass flex items-center justify-between gap-4 border-warn/40 bg-warn/8 p-4"
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

      {/* ---- SECTION: what's happening now ---- */}
      <Panel eyebrow="Live" title="What's happening now">
        <NowRunning active={activeStage} />
        <div className="mt-5">
          <PhaseTimeline stages={stages} />
        </div>
      </Panel>

      {/* ---- SECTION: problem definition + activity ---- */}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-[0.9fr_1.1fr]">
        <Panel eyebrow="Step 1 · confirmed" title="The problem being solved">
          {spec ? (
            <div className="space-y-3">
              <Field label="Dataset" value={run?.dataset_name ?? "—"} mono />
              <Field label="Target column" value={spec.target_column} mono />
              <Field label="Task type" value={spec.task_type.replace(/_/g, " ")} />
              <Field label="Primary metric" value={spec.primary_metric} mono />
              {spec.cost_asymmetry && spec.cost_asymmetry.false_negative_cost !== spec.cost_asymmetry.false_positive_cost && (
                <Field
                  label="Error cost"
                  value={`FN ${spec.cost_asymmetry.false_negative_cost} : FP ${spec.cost_asymmetry.false_positive_cost}`}
                />
              )}
              {spec.assumptions.length > 0 && (
                <div>
                  <div className="label-eyebrow mb-1.5">Assumptions</div>
                  <ul className="space-y-1 text-sm text-ink-muted">
                    {spec.assumptions.map((a, i) => (
                      <li key={i} className="flex gap-2">
                        <span className="text-accent-bright">•</span>
                        {a}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ) : (
            <EmptyState icon="◈" title="Problem not confirmed yet" hint="The Requirement agent's spec appears here once Gate 1 is reached." />
          )}
        </Panel>

        <Panel eyebrow="Live" title="Activity feed" className="pb-4">
          <ActivityFeed events={stream.events} height={360} />
        </Panel>
      </div>

      {/* ---- SECTION: models being trained ---- */}
      <Panel eyebrow="Experiments" title="Models on the leaderboard">
        <LeaderboardMini runId={runId} metric={metric} />
      </Panel>

      {/* ---- SECTION: artifact store + lineage (explained) ---- */}
      <Panel
        eyebrow="Provenance"
        title="Artifact store & lineage"
        action={
          <Link href={`/lineage?run=${runId}`} className="btn-ghost text-xs">
            Open lineage graph
          </Link>
        }
      >
        <p className="-mt-2 mb-4 max-w-3xl text-sm text-ink-muted">
          Every decision the team makes — each profile, split, cleaning step, trained model, and report — is written to a
          content-addressed store. Click any artifact to see exactly what it is and its real contents.
        </p>
        <ArtifactStore runId={runId} />
      </Panel>

      {/* ---- SECTION: system map + verdict ---- */}
      <div className="grid grid-cols-1 gap-6 xl:grid-cols-[1.15fr_0.85fr]">
        <Panel eyebrow="System map" title="Agent flow">
          <AgentFlow stages={stages} />
          <div className="mt-3 flex flex-wrap justify-center gap-4 text-xs text-ink-faint">
            <Legend color="#4f46e5" label="active" />
            <Legend color="#059669" label="done" />
            <Legend color="#d97706" label="gate / awaiting" />
            <Legend color="#c3c9d5" label="skipped" />
            <Legend color="#cbd2de" label="pending" />
          </div>
        </Panel>

        <Panel eyebrow="Outcome" title="Critic verdict">
          {critique ? (
            <div>
              <VerdictBadge verdict={critique.verdict} />
              {critique.findings.length > 0 && (
                <ul className="mt-3 space-y-2 text-sm">
                  {critique.findings.map((f, i) => (
                    <li key={i} className="rounded-lg border border-base-600 bg-base-800 p-2.5">
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] font-bold uppercase tracking-wide text-accent-bright">{f.severity}</span>
                        <span className="text-[11px] text-ink-faint">{f.category}</span>
                      </div>
                      <p className="mt-1 text-ink-muted">{f.evidence}</p>
                      {f.recommendation && <p className="mt-1 text-ink-faint">→ {f.recommendation}</p>}
                    </li>
                  ))}
                </ul>
              )}
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
            </div>
          ) : (
            <EmptyState icon="⚖" title="No verdict yet" hint="The Critic's judgment appears here once evaluation completes." />
          )}
        </Panel>
      </div>
    </div>
  );
}

function Field({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-base-600/60 pb-2">
      <span className="label-eyebrow">{label}</span>
      <span className={`text-sm text-ink ${mono ? "font-mono" : ""}`}>{value}</span>
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
