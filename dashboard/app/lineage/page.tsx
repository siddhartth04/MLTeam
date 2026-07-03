"use client";

import { motion } from "framer-motion";
import { useMemo, useState } from "react";

import { useActiveRun } from "@/components/RunContext";
import { EmptyState, Panel, SectionTitle } from "@/components/primitives";
import { useArtifact, useLineage } from "@/lib/api";
import type { LineageGraph, LineageNode } from "@/lib/types";

const TYPE_COLOR: Record<string, string> = {
  ProblemSpec: "#7c5cff",
  DatasetVersion: "#31d0ff",
  DataProfile: "#38e8a0",
  SplitPlan: "#ffb648",
  CleaningManifest: "#9d84ff",
  FeaturePlan: "#5eead4",
  ExperimentPlan: "#f0a6ff",
  TrainedModel: "#38e8a0",
  EvaluationReport: "#31d0ff",
  ExplainabilityReport: "#ffd166",
  CritiqueReport: "#ff5d73",
  DeploymentBundle: "#a0f0c0",
  FinalReport: "#7c5cff",
};

interface Placed extends LineageNode {
  x: number;
  y: number;
  depth: number;
}

function assignDepths(graph: LineageGraph): Map<string, number> {
  const parents = new Map<string, string[]>();
  graph.nodes.forEach((n) => parents.set(n.id, []));
  graph.edges.forEach((e) => parents.get(e.target)?.push(e.source));
  const depth = new Map<string, number>();
  const visit = (id: string, seen: Set<string>): number => {
    if (depth.has(id)) return depth.get(id)!;
    if (seen.has(id)) return 0;
    seen.add(id);
    const ps = parents.get(id) ?? [];
    const d = ps.length === 0 ? 0 : 1 + Math.max(...ps.map((p) => visit(p, seen)));
    depth.set(id, d);
    return d;
  };
  graph.nodes.forEach((n) => visit(n.id, new Set()));
  return depth;
}

export default function LineagePage() {
  const { runId } = useActiveRun();
  const { data: graph } = useLineage(runId);
  const [selected, setSelected] = useState<string | null>(null);
  const { data: detail } = useArtifact(selected);

  const placed = useMemo<Placed[]>(() => {
    if (!graph) return [];
    const depth = assignDepths(graph);
    const byDepth = new Map<number, LineageNode[]>();
    graph.nodes.forEach((n) => {
      const d = depth.get(n.id) ?? 0;
      byDepth.set(d, [...(byDepth.get(d) ?? []), n]);
    });
    const colW = 210;
    const rowH = 74;
    const out: Placed[] = [];
    Array.from(byDepth.entries())
      .sort((a, b) => a[0] - b[0])
      .forEach(([d, nodes]) => {
        nodes.forEach((n, i) => {
          out.push({ ...n, depth: d, x: 40 + d * colW, y: 40 + i * rowH });
        });
      });
    return out;
  }, [graph]);

  const width = Math.max(700, (Math.max(0, ...placed.map((p) => p.depth)) + 1) * 210 + 60);
  const height = Math.max(360, ...placed.map((p) => p.y + 60), 360);
  const posById = new Map(placed.map((p) => [p.id, p]));

  if (!runId) {
    return (
      <Panel>
        <EmptyState icon="❖" title="No run selected" hint="Pick a run to explore its artifact lineage." />
      </Panel>
    );
  }

  return (
    <div>
      <SectionTitle sub="Every artifact and who produced it — the content-addressed provenance graph. Click a node to inspect it.">
        Artifact lineage
      </SectionTitle>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-[1.4fr_0.6fr]">
        <Panel eyebrow="Provenance DAG" title="Flow of artifacts">
          {placed.length === 0 ? (
            <EmptyState icon="❖" title="No artifacts yet" hint="They appear as the run produces them." />
          ) : (
            <div className="overflow-auto rounded-xl border border-base-600/50 bg-base-900/40" style={{ maxHeight: 560 }}>
              <svg width={width} height={height} className="min-w-full">
                {graph!.edges.map((e, i) => {
                  const s = posById.get(e.source);
                  const t = posById.get(e.target);
                  if (!s || !t) return null;
                  const x1 = s.x + 170;
                  const y1 = s.y + 20;
                  const x2 = t.x;
                  const y2 = t.y + 20;
                  const mx = (x1 + x2) / 2;
                  return (
                    <path
                      key={i}
                      d={`M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`}
                      fill="none"
                      stroke="#2a345c"
                      strokeWidth={1.3}
                    />
                  );
                })}
                {placed.map((n) => {
                  const color = TYPE_COLOR[n.type] ?? "#9aa6c8";
                  const active = n.id === selected;
                  return (
                    <g
                      key={n.id}
                      transform={`translate(${n.x},${n.y})`}
                      onClick={() => setSelected(n.id)}
                      style={{ cursor: "pointer" }}
                    >
                      <rect
                        width={170}
                        height={40}
                        rx={10}
                        fill={active ? "#1a2138" : "#111730"}
                        stroke={active ? color : `${color}66`}
                        strokeWidth={active ? 2 : 1.3}
                      />
                      <rect width={4} height={40} rx={2} fill={color} />
                      <text x={14} y={17} fontSize="11" fontWeight="700" fill="#e8ecf8">
                        {n.type}
                        {n.protected ? "  🔒" : ""}
                      </text>
                      <text x={14} y={31} fontSize="9" fill="#5f6b8f" fontFamily="monospace">
                        {n.producer} · {n.short_id}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>
          )}
        </Panel>

        <Panel eyebrow="Inspector" title={detail ? detail.type : "Select an artifact"}>
          {!detail ? (
            <EmptyState icon="◌" title="Nothing selected" hint="Click a node in the graph." />
          ) : (
            <motion.div key={detail.id} initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-3">
              <Row label="ID" value={<span className="mono">{detail.short_id}…</span>} />
              <Row label="Producer" value={detail.producer} />
              <Row label="Config hash" value={<span className="mono">{detail.config_hash || "—"}</span>} />
              <Row label="Parents" value={String(detail.parents.length)} />
              <Row label="Children" value={String(detail.children.length)} />
              {detail.protected && (
                <div className="rounded-lg border border-warn/40 bg-warn/10 p-2.5 text-xs text-warn">
                  🔒 Protected holdout — content withheld until final reporting.
                </div>
              )}
              <div>
                <div className="label-eyebrow mb-1">Payload</div>
                <pre className="max-h-72 overflow-auto rounded-lg border border-base-600/50 bg-base-900/60 p-3 text-[11px] text-ink-muted">
                  {JSON.stringify(detail.payload, null, 2)}
                </pre>
              </div>
            </motion.div>
          )}
        </Panel>
      </div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between border-b border-base-700/40 pb-2 text-sm">
      <span className="text-ink-faint">{label}</span>
      <span className="text-ink">{value}</span>
    </div>
  );
}
