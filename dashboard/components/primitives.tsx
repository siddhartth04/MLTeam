"use client";

import { motion } from "framer-motion";
import type { ReactNode } from "react";

import type { RunStatus, StageStatus } from "@/lib/types";

export function Panel({
  children,
  className = "",
  title,
  eyebrow,
  action,
}: {
  children: ReactNode;
  className?: string;
  title?: string;
  eyebrow?: string;
  action?: ReactNode;
}) {
  return (
    <section className={`glass p-5 ${className}`}>
      {(title || eyebrow || action) && (
        <header className="mb-4 flex items-start justify-between gap-3">
          <div>
            {eyebrow && <div className="label-eyebrow mb-1">{eyebrow}</div>}
            {title && <h3 className="text-lg font-semibold text-ink">{title}</h3>}
          </div>
          {action}
        </header>
      )}
      {children}
    </section>
  );
}

export function StatTile({
  label,
  value,
  sub,
  accent = "accent",
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  accent?: "accent" | "ok" | "warn" | "cyan" | "danger";
}) {
  const bar: Record<string, string> = {
    accent: "from-accent to-accent-dim",
    ok: "from-ok to-ok-dim",
    warn: "from-warn to-warn-dim",
    cyan: "from-cyan to-cyan-dim",
    danger: "from-danger to-danger-dim",
  };
  return (
    <div className="glass-raised relative overflow-hidden p-4">
      <div className={`absolute left-0 top-0 h-full w-1 bg-gradient-to-b ${bar[accent]}`} />
      <div className="label-eyebrow">{label}</div>
      <div className="stat-value mt-1">{value}</div>
      {sub && <div className="mt-1 text-xs text-ink-muted">{sub}</div>}
    </div>
  );
}

const STATUS_STYLE: Record<string, { bg: string; text: string; label: string; dot: string }> = {
  completed: { bg: "bg-ok/10 border-ok/40", text: "text-ok", label: "Completed", dot: "bg-ok" },
  running: { bg: "bg-accent/10 border-accent/40", text: "text-accent-bright", label: "Running", dot: "bg-accent-bright" },
  awaiting_approval: { bg: "bg-warn/10 border-warn/40", text: "text-warn", label: "Awaiting approval", dot: "bg-warn" },
  queued: { bg: "bg-base-600/40 border-base-500/60", text: "text-ink-muted", label: "Queued", dot: "bg-ink-faint" },
  rejected: { bg: "bg-danger/10 border-danger/40", text: "text-danger", label: "Rejected", dot: "bg-danger" },
  failed: { bg: "bg-danger/10 border-danger/40", text: "text-danger", label: "Failed", dot: "bg-danger" },
};

export function StatusPill({ status }: { status: RunStatus | string }) {
  const s = STATUS_STYLE[status] ?? STATUS_STYLE.queued;
  const live = status === "running" || status === "awaiting_approval";
  return (
    <span
      className={`chip border ${s.bg} ${s.text} font-semibold`}
      style={{ borderStyle: "solid" }}
    >
      <span className={`relative h-2 w-2 rounded-full ${s.dot}`}>
        {live && (
          <span className={`absolute inset-0 rounded-full ${s.dot} animate-ping opacity-75`} />
        )}
      </span>
      {s.label}
    </span>
  );
}

const VERDICT_STYLE: Record<string, string> = {
  approve: "bg-ok/15 text-ok border-ok/40",
  iterate_with_actions: "bg-warn/15 text-warn border-warn/40",
  halt: "bg-danger/15 text-danger border-danger/40",
};

export function VerdictBadge({ verdict }: { verdict: string }) {
  return (
    <span
      className={`inline-flex items-center rounded-lg border px-3 py-1 text-sm font-bold ${
        VERDICT_STYLE[verdict] ?? "border-base-500 text-ink-muted"
      }`}
    >
      {verdict.replace(/_/g, " ")}
    </span>
  );
}

export function PassFail({ passed }: { passed: boolean }) {
  return passed ? (
    <span className="font-semibold text-ok">✓ PASS</span>
  ) : (
    <span className="font-semibold text-danger">✕ FAIL</span>
  );
}

export function SectionTitle({ children, sub }: { children: ReactNode; sub?: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      className="mb-6"
    >
      <h1 className="bg-gradient-to-r from-ink to-ink-muted bg-clip-text text-2xl font-bold tracking-tight text-transparent">
        {children}
      </h1>
      {sub && <p className="mt-1 text-sm text-ink-muted">{sub}</p>}
    </motion.div>
  );
}

export function EmptyState({ icon, title, hint }: { icon: string; title: string; hint?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-16 text-center">
      <div className="text-4xl text-ink-faint opacity-60">{icon}</div>
      <div className="text-ink-muted">{title}</div>
      {hint && <div className="max-w-md text-sm text-ink-faint">{hint}</div>}
    </div>
  );
}

export const STAGE_COLOR: Record<StageStatus, string> = {
  pending: "#39415f",
  active: "#7c5cff",
  done: "#38e8a0",
  awaiting_approval: "#ffb648",
  rejected: "#ff5d73",
  skipped: "#5f6b8f",
  failed: "#ff5d73",
};
