"use client";

// Streaming activity feed: a live terminal-style log of trace events with
// level-coded accents and auto-scroll to the newest line.

import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef } from "react";

import { STAGE_BY_KEY } from "@/lib/stages";
import type { LogEvent } from "@/lib/types";

const LEVEL: Record<string, { dot: string; text: string }> = {
  info: { dot: "bg-ink-faint", text: "text-ink-muted" },
  success: { dot: "bg-ok", text: "text-ink" },
  warning: { dot: "bg-warn", text: "text-warn" },
  error: { dot: "bg-danger", text: "text-danger" },
  gate: { dot: "bg-accent-bright", text: "text-accent-bright" },
};

function time(ts: string): string {
  try {
    return new Date(ts).toLocaleTimeString([], { hour12: false });
  } catch {
    return "";
  }
}

export function ActivityFeed({ events, height = 360 }: { events: LogEvent[]; height?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    ref.current?.scrollTo({ top: ref.current.scrollHeight, behavior: "smooth" });
  }, [events.length]);

  const recent = events.slice(-160);

  return (
    <div
      ref={ref}
      className="overflow-y-auto rounded-xl border border-base-600/50 bg-base-900/50 p-3 font-mono text-[12.5px]"
      style={{ height }}
    >
      {recent.length === 0 && (
        <div className="grid h-full place-items-center text-ink-faint">
          Waiting for activity…
        </div>
      )}
      <AnimatePresence initial={false}>
        {recent.map((ev, i) => {
          const style = LEVEL[ev.level] ?? LEVEL.info;
          const stage = ev.stage ? STAGE_BY_KEY[ev.stage]?.short : null;
          return (
            <motion.div
              key={`${ev.timestamp}-${i}`}
              initial={{ opacity: 0, x: -6 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-start gap-2.5 py-[3px]"
            >
              <span className="mt-1.5 flex-none">
                <span className={`block h-1.5 w-1.5 rounded-full ${style.dot}`} />
              </span>
              <span className="flex-none text-ink-faint">{time(ev.timestamp)}</span>
              {stage && (
                <span className="flex-none rounded bg-base-700/70 px-1.5 text-[10px] text-ink-faint">
                  {stage}
                </span>
              )}
              <span className={`${style.text} break-words`}>{ev.message}</span>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
