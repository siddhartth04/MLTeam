"use client";

import { motion } from "framer-motion";
import { useMemo, useState } from "react";

import { useActiveRun } from "@/components/RunContext";
import { EmptyState, Panel, SectionTitle, StatTile } from "@/components/primitives";
import { useRun, useRunEvents } from "@/lib/api";
import { STAGE_BY_KEY } from "@/lib/stages";
import type { LogEvent } from "@/lib/types";
import { useRunStream } from "@/lib/useRunStream";

const LEVELS = ["all", "gate", "success", "warning", "error", "info"] as const;

const LEVEL_STYLE: Record<string, { border: string; badge: string; text: string }> = {
  info: { border: "border-base-600/60", badge: "bg-base-700 text-ink-faint", text: "text-ink-muted" },
  success: { border: "border-ok/40", badge: "bg-ok/15 text-ok", text: "text-ink" },
  warning: { border: "border-warn/40", badge: "bg-warn/15 text-warn", text: "text-warn" },
  error: { border: "border-danger/40", badge: "bg-danger/15 text-danger", text: "text-danger" },
  gate: { border: "border-accent/40", badge: "bg-accent/15 text-accent-bright", text: "text-accent-bright" },
};

export default function TracePage() {
  const { runId } = useActiveRun();
  const { data: run } = useRun(runId, 4000);
  const live = run?.status === "running" || run?.status === "awaiting_approval";
  const stream = useRunStream(live ? runId : null);
  const { data: polled } = useRunEvents(runId, live ? 0 : 0);
  const [filter, setFilter] = useState<(typeof LEVELS)[number]>("all");

  const events = useMemo<LogEvent[]>(() => {
    return stream.events.length ? stream.events : (polled ?? []);
  }, [stream.events, polled]);

  const filtered = filter === "all" ? events : events.filter((e) => e.level === filter);
  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    events.forEach((e) => (c[e.level] = (c[e.level] ?? 0) + 1));
    return c;
  }, [events]);

  if (!runId) {
    return (
      <Panel>
        <EmptyState icon="≣" title="No run selected" hint="Pick a run to inspect its full trace." />
      </Panel>
    );
  }

  return (
    <div>
      <SectionTitle sub="The complete audit log — every agent decision, gate, and event, in order.">
        Run trace
      </SectionTitle>

      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <StatTile label="Total events" value={String(events.length)} accent="cyan" />
        <StatTile label="Gate events" value={String(counts.gate ?? 0)} accent="accent" />
        <StatTile label="Warnings" value={String(counts.warning ?? 0)} accent="warn" />
        <StatTile label="Errors" value={String(counts.error ?? 0)} accent={counts.error ? "danger" : "ok"} />
      </div>

      <Panel eyebrow="Timeline" title="Events" className="mt-6"
        action={
          <div className="flex flex-wrap gap-1.5">
            {LEVELS.map((l) => (
              <button
                key={l}
                onClick={() => setFilter(l)}
                className={`rounded-lg px-2.5 py-1 text-xs font-medium capitalize transition-colors ${
                  filter === l ? "bg-accent/15 text-accent-bright" : "text-ink-faint hover:text-ink"
                }`}
              >
                {l}
                {l !== "all" && counts[l] ? ` (${counts[l]})` : ""}
              </button>
            ))}
          </div>
        }
      >
        {filtered.length === 0 ? (
          <EmptyState icon="≣" title="No events" hint="Nothing at this filter level." />
        ) : (
          <div className="relative space-y-2 pl-4">
            <div className="absolute bottom-2 left-[7px] top-2 w-px bg-base-600/60" />
            {filtered.map((ev, i) => {
              const style = LEVEL_STYLE[ev.level] ?? LEVEL_STYLE.info;
              const stage = ev.stage ? STAGE_BY_KEY[ev.stage] : null;
              const tokensIn = ev.extra?.tokens_in;
              return (
                <motion.div
                  key={`${ev.timestamp}-${i}`}
                  initial={{ opacity: 0, x: -4 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: Math.min(i * 0.01, 0.3) }}
                  className={`relative rounded-xl border bg-base-800/50 p-3 ${style.border}`}
                >
                  <span className={`absolute -left-[13px] top-4 h-2.5 w-2.5 rounded-full ring-2 ring-base-900 ${style.badge.split(" ")[0]}`} />
                  <div className="flex flex-wrap items-center gap-2">
                    <span className={`rounded px-1.5 py-0.5 text-[10px] font-bold uppercase ${style.badge}`}>
                      {ev.level}
                    </span>
                    {stage && (
                      <span className="chip !py-0.5 !text-[10px]">{stage.short}</span>
                    )}
                    <span className="mono">{new Date(ev.timestamp).toLocaleTimeString([], { hour12: false })}</span>
                    {typeof tokensIn === "number" && (
                      <span className="mono text-ink-faint">
                        {tokensIn}↓ / {String(ev.extra?.tokens_out ?? 0)}↑ tok
                      </span>
                    )}
                  </div>
                  <div className={`mt-1.5 text-sm ${style.text}`}>{ev.message}</div>
                </motion.div>
              );
            })}
          </div>
        )}
      </Panel>
    </div>
  );
}
