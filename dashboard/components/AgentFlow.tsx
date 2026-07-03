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
  pending: { ring: "#2b3352", fill: "#141a30", glow: "transparent" },
  active: { ring: "#7c5cff", fill: "#221a44", glow: "#7c5cff" },
  done: { ring: "#38e8a0", fill: "#10261d", glow: "#38e8a0" },
  awaiting_approval: { ring: "#ffb648", fill: "#2a2110", glow: "#ffb648" },
  rejected: { ring: "#ff5d73", fill: "#2a1116", glow: "#ff5d73" },
  skipped: { ring: "#4a557a", fill: "#12172a", glow: "transparent" },
  failed: { ring: "#ff5d73", fill: "#2a1116", glow: "#ff5d73" },
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
            <stop offset="0%" stopColor="#7c5cff" stopOpacity="0.5" />
            <stop offset="60%" stopColor="#31d0ff" stopOpacity="0.12" />
            <stop offset="100%" stopColor="#31d0ff" stopOpacity="0" />
          </radialGradient>
          <linearGradient id="edgeGrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#7c5cff" stopOpacity="0.05" />
            <stop offset="100%" stopColor="#7c5cff" stopOpacity="0.5" />
          </linearGradient>
        </defs>

        {/* orbit guide */}
        <circle cx={C} cy={C} r={R} fill="none" stroke="#232c49" strokeWidth={1} strokeDasharray="2 6" />

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
              stroke={activeEdge ? "url(#edgeGrad)" : doneEdge ? "#38e8a033" : "#1c2440"}
              strokeWidth={activeEdge ? 2.5 : 1.2}
              strokeDasharray={activeEdge ? "5 7" : undefined}
              className={activeEdge ? "animate-flow-dash" : undefined}
            />
          );
        })}

        {/* central artifact store */}
        <circle cx={C} cy={C} r={92} fill="url(#hubGrad)" className={anyActive ? "animate-breathe" : ""} />
        <circle cx={C} cy={C} r={54} fill="#0d1223" stroke="#33407a" strokeWidth={1.5} />
        <circle cx={C} cy={C} r={54} fill="none" stroke="#7c5cff" strokeWidth={1} opacity={0.35} />
        <text x={C} y={C - 6} textAnchor="middle" className="fill-ink" fontSize="15" fontWeight="700">
          ARTIFACT
        </text>
        <text x={C} y={C + 13} textAnchor="middle" className="fill-ink" fontSize="15" fontWeight="700">
          STORE
        </text>
        <text x={C} y={C + 32} textAnchor="middle" fill="#5f6b8f" fontSize="9" letterSpacing="2">
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
                fill={st === "pending" ? "#5f6b8f" : "#e8ecf8"}
              >
                {node.icon}
              </text>
              {/* label chip below */}
              <text
                textAnchor="middle"
                dy={NODE_R + 15}
                fontSize="10"
                fontWeight="600"
                fill={st === "pending" ? "#5f6b8f" : st === "active" ? "#9d84ff" : st === "done" ? "#38e8a0" : "#9aa6c8"}
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
