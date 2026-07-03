"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Suspense, type ReactNode } from "react";

import { useGates, useRuns } from "@/lib/api";

import { RunProvider, useActiveRun } from "./RunContext";
import { StatusPill } from "./primitives";

const NAV = [
  { href: "/", label: "Launch", icon: "◆" },
  { href: "/mission", label: "Mission Control", icon: "◉" },
  { href: "/gates", label: "Approval Gates", icon: "⛴" },
  { href: "/experiments", label: "Experiments", icon: "▤" },
  { href: "/lineage", label: "Lineage", icon: "❖" },
  { href: "/trace", label: "Trace", icon: "≣" },
  { href: "/report", label: "Report", icon: "▦" },
];

function RunSelector() {
  const { data: runs } = useRuns();
  const { runId, setRunId } = useActiveRun();
  const list = runs ?? [];
  return (
    <div className="px-3">
      <div className="label-eyebrow mb-2 px-1">Active run</div>
      <div className="max-h-52 space-y-1 overflow-y-auto pr-1">
        {list.length === 0 && (
          <div className="px-2 py-3 text-xs text-ink-faint">No runs yet — launch one.</div>
        )}
        {list.map((r) => {
          const active = r.ui_run_id === runId;
          return (
            <button
              key={r.ui_run_id}
              onClick={() => setRunId(r.ui_run_id)}
              className={`w-full rounded-xl border px-3 py-2 text-left transition-all ${
                active
                  ? "border-accent/60 bg-accent/10"
                  : "border-transparent bg-base-700/40 hover:bg-base-600/50"
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="mono truncate text-ink">{r.ui_run_id}</span>
                <span
                  className={`h-2 w-2 flex-none rounded-full ${
                    r.status === "running" || r.status === "awaiting_approval"
                      ? "bg-accent-bright"
                      : r.status === "completed"
                        ? "bg-ok"
                        : r.status === "failed" || r.status === "rejected"
                          ? "bg-danger"
                          : "bg-ink-faint"
                  }`}
                />
              </div>
              <div className="mt-0.5 truncate text-[11px] text-ink-faint">{r.goal_text}</div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function GateBadge() {
  const { data: gates } = useGates();
  const count = gates?.length ?? 0;
  if (count === 0) return null;
  return (
    <span className="ml-auto inline-flex h-5 min-w-5 items-center justify-center rounded-full bg-warn px-1.5 text-[11px] font-bold text-base-900">
      {count}
    </span>
  );
}

function TopBar() {
  const { runId } = useActiveRun();
  const { data: runs } = useRuns();
  const run = runs?.find((r) => r.ui_run_id === runId);
  return (
    <div className="sticky top-0 z-20 flex items-center gap-4 border-b border-base-600/60 bg-base-900/70 px-8 py-4 backdrop-blur-xl">
      <div className="flex items-center gap-3">
        <span className="mono text-ink-faint">RUN</span>
        <span className="mono text-ink">{runId ?? "— none selected —"}</span>
      </div>
      {run && (
        <>
          <span className="text-ink-faint">·</span>
          <span className="max-w-md truncate text-sm text-ink-muted">{run.goal_text}</span>
          <div className="ml-auto flex items-center gap-4">
            <span className="mono text-ink-muted">
              LLM ${run.llm_cost_usd.toFixed(4)}
            </span>
            <StatusPill status={run.status} />
          </div>
        </>
      )}
    </div>
  );
}

function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="fixed left-0 top-0 z-30 flex h-screen w-64 flex-col gap-6 border-r border-base-600/60 bg-base-850/80 py-6 backdrop-blur-xl">
      <div className="flex items-center gap-3 px-5">
        <div className="grid h-10 w-10 place-items-center rounded-xl bg-accent-grad shadow-glow shadow-accent/60">
          <span className="text-lg font-black text-white">M</span>
        </div>
        <div>
          <div className="text-sm font-bold leading-tight text-ink">Mission Control</div>
          <div className="text-[10px] uppercase tracking-[0.2em] text-ink-faint">
            ML Engineering Team
          </div>
        </div>
      </div>

      <nav className="flex flex-col gap-1 px-3">
        {NAV.map((item) => {
          const active = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`group relative flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-all ${
                active ? "bg-accent/12 text-ink" : "text-ink-muted hover:bg-base-700/50 hover:text-ink"
              }`}
            >
              {active && (
                <motion.span
                  layoutId="nav-active"
                  className="absolute left-0 top-1/2 h-6 w-1 -translate-y-1/2 rounded-full bg-accent-grad"
                />
              )}
              <span className={`text-base ${active ? "text-accent-bright" : "text-ink-faint"}`}>
                {item.icon}
              </span>
              {item.label}
              {item.href === "/gates" && <GateBadge />}
            </Link>
          );
        })}
      </nav>

      <div className="mt-2 border-t border-base-600/50 pt-4">
        <RunSelector />
      </div>

      <div className="mt-auto px-5">
        <div className="glass-raised p-3">
          <div className="text-[11px] leading-relaxed text-ink-faint">
            Agents decide, libraries execute. Holdout frozen at ingestion — read only at report time.
          </div>
        </div>
      </div>
    </aside>
  );
}

export function Shell({ children }: { children: ReactNode }) {
  return (
    <Suspense fallback={<div className="p-8 text-ink-muted">Loading…</div>}>
      <RunProvider>
        <Sidebar />
        <div className="ml-64 flex min-h-screen flex-col">
          <TopBar />
          <main className="flex-1 px-8 py-8">{children}</main>
        </div>
      </RunProvider>
    </Suspense>
  );
}
