"use client";

// The centerpiece: an animated hub-and-spoke X-ray of the agentic system.
// Agents orbit a central artifact store. Each node reflects live stage status —
// pulsing while active, glowing green when done, amber at gates. Edges pulse as
// artifacts flow toward / from the store. Pure SVG + Framer Motion.

import { AnimatePresence, motion } from "framer-motion";
import { useMemo } from "react";

import { STAGES } from "@/lib/stages";
import type { StageState, StageStatus } from "@/lib/types";

const SIZE = 560;
const C = SIZE / 2;
const R = 218; // orbit radius
const NODE_R = 30;

const COLORS: Record<StageStatus, { ring: string; fill: string; glow: string }> = {
  pending: { ring: "#cbd2de", fill: "#f3f4f8", glow: "transparent" },
  active: { ring: "#4f46e5", fill: "#eef0fe", glow: "#6366f1" },
  done: { ring: "#059669", fill: "#e7f7f0", glow: "#10b981" },
  awaiting_approval: { ring: "#d97706", fill: "#fdf4e5", glow: "#f59e0b" },
  rejected: { ring: "#dc2626", fill: "#fdecec", glow: "#ef4444" },
  skipped: { ring: "#c3c9d5", fill: "#f4f5f8", glow: "transparent" },
  failed: { ring: "#dc2626", fill: "#fdecec", glow: "#ef4444" },
};

interface NodePos {
  key: string;
  label: string;
  short: string;
  isGate: boolean;
  icon: string;
  x: number;
  y: number;
}

function layout(): NodePos[] {
  const n = STAGES.length;
  return STAGES.map((s, i) => {
    const angle = (i / n) * Math.PI * 2 - Math.PI / 2;
    return {
      key: s.key,
      label: s.label,
      short: s.short,
      isGate: s.isGate,
      icon: s.icon,
      x: C + R * Math.cos(angle),
      y: C + R * Math.sin(angle),
    };
  });
}

export function AgentFlow({
  stages,
  onSelect,
  activeKey,
}: {
  stages: StageState[];
  onSelect?: (key: string) => void;
  activeKey?: string | null;
}) {
  const nodes = useMemo(layout, []);
  const statusOf = useMemo(() => {
    const m = new Map(stages.map((s) => [s.stage, s.status]));
    return (key: string): StageStatus => (m.get(key) as StageStatus) ?? "pending";
  }, [stages]);

  const anyActive = stages.some((s) => s.status === "active");

  return (
    <div className="relative mx-auto aspect-square w-full max-w-[560px]">
      <svg viewBox={`0 0 ${SIZE} ${SIZE}`} className="h-full w-full">
        <defs>
          <radialGradient id="hubGrad" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#6366f1" stopOpacity="0.16" />
            <stop offset="60%" stopColor="#22d3ee" stopOpacity="0.06" />
            <stop offset="100%" stopColor="#22d3ee" stopOpacity="0" />
          </radialGradient>
          <linearGradient id="edgeGrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#6366f1" stopOpacity="0.08" />
            <stop offset="100%" stopColor="#4f46e5" stopOpacity="0.55" />
          </linearGradient>
        </defs>

        {/* orbit guide */}
        <circle cx={C} cy={C} r={R} fill="none" stroke="#dde1e9" strokeWidth={1} strokeDasharray="2 6" />

        {/* edges: store <-> node */}
        {nodes.map((node) => {
          const st = statusOf(node.key);
          const activeEdge = st === "active";
          const doneEdge = st === "done";
          return (
            <line
              key={`e-${node.key}`}
              x1={C}
              y1={C}
              x2={node.x}
              y2={node.y}
              stroke={activeEdge ? "url(#edgeGrad)" : doneEdge ? "#05966955" : "#e4e7ec"}
              strokeWidth={activeEdge ? 2.5 : 1.2}
              strokeDasharray={activeEdge ? "5 7" : undefined}
              className={activeEdge ? "animate-flow-dash" : undefined}
            />
          );
        })}

        {/* central artifact store */}
        <circle cx={C} cy={C} r={92} fill="url(#hubGrad)" className={anyActive ? "animate-breathe" : ""} />
        <circle cx={C} cy={C} r={54} fill="#ffffff" stroke="#c7cbf5" strokeWidth={1.5} />
        <circle cx={C} cy={C} r={54} fill="none" stroke="#6366f1" strokeWidth={1} opacity={0.4} />
        <text x={C} y={C - 6} textAnchor="middle" className="fill-ink" fontSize="15" fontWeight="700">
          ARTIFACT
        </text>
        <text x={C} y={C + 13} textAnchor="middle" className="fill-ink" fontSize="15" fontWeight="700">
          STORE
        </text>
        <text x={C} y={C + 32} textAnchor="middle" fill="#98a1b0" fontSize="9" letterSpacing="2">
          HUB · CONTENT-ADDRESSED
        </text>

        {/* nodes */}
        {nodes.map((node, i) => {
          const st = statusOf(node.key);
          const c = COLORS[st] ?? COLORS.pending;
          const selected = activeKey === node.key;
          return (
            <g
              key={node.key}
              transform={`translate(${node.x},${node.y})`}
              onClick={() => onSelect?.(node.key)}
              style={{ cursor: onSelect ? "pointer" : "default" }}
            >
              {/* pulse ring for active */}
              {st === "active" && (
                <circle r={NODE_R} fill="none" stroke={c.ring} strokeWidth={2} className="animate-pulse-ring" />
              )}
              {st === "awaiting_approval" && (
                <circle r={NODE_R} fill="none" stroke={c.ring} strokeWidth={2} className="animate-pulse-ring" />
              )}
              <motion.circle
                r={NODE_R}
                fill={c.fill}
                stroke={c.ring}
                strokeWidth={node.isGate ? 2.5 : selected ? 2.5 : 1.6}
                initial={{ scale: 0 }}
                animate={{
                  scale: 1,
                  filter: c.glow !== "transparent" ? `drop-shadow(0 0 10px ${c.glow})` : "none",
                }}
                transition={{ delay: i * 0.02, type: "spring", stiffness: 200, damping: 18 }}
                strokeDasharray={node.isGate ? "3 3" : undefined}
              />
              <text
                textAnchor="middle"
                dy="6"
                fontSize="18"
                fill={st === "pending" ? "#98a1b0" : "#1a2231"}
              >
                {node.icon}
              </text>
              {/* label chip below */}
              <text
                textAnchor="middle"
                dy={NODE_R + 15}
                fontSize="10"
                fontWeight="600"
                fill={st === "pending" ? "#98a1b0" : st === "active" ? "#4338ca" : st === "done" ? "#059669" : "#586173"}
              >
                {node.short}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
