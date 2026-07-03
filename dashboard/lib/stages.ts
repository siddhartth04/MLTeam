// Stage metadata shared by the flow diagram and the stage rail. Mirrors
// mlteam.api.events.Stage ordering.

export interface StageMeta {
  key: string;
  label: string;
  short: string;
  isGate: boolean;
  icon: string; // single glyph for the node
}

export const STAGES: StageMeta[] = [
  { key: "requirement", label: "Requirement Understanding", short: "Requirement", isGate: false, icon: "◈" },
  { key: "gate_spec", label: "Gate 1 · Problem Spec", short: "Gate 1", isGate: true, icon: "⛴" },
  { key: "ingest", label: "Dataset Ingestion", short: "Ingest", isGate: false, icon: "⬇" },
  { key: "profile", label: "Profiling", short: "Profile", isGate: false, icon: "◔" },
  { key: "split", label: "Split Strategist", short: "Split", isGate: false, icon: "⑂" },
  { key: "clean", label: "Data Cleaning", short: "Clean", isGate: false, icon: "❋" },
  { key: "features", label: "Feature Engineering", short: "Features", isGate: false, icon: "✦" },
  { key: "plan", label: "Experiment Planner", short: "Plan", isGate: false, icon: "❑" },
  { key: "gate_plan", label: "Gate 2 · Training Plan", short: "Gate 2", isGate: true, icon: "⛴" },
  { key: "train", label: "Training", short: "Train", isGate: false, icon: "⚙" },
  { key: "evaluate", label: "Evaluation", short: "Evaluate", isGate: false, icon: "▤" },
  { key: "critic", label: "Critic", short: "Critic", isGate: false, icon: "⚖" },
  { key: "iterate", label: "Iteration Loop", short: "Iterate", isGate: false, icon: "↻" },
  { key: "explain", label: "Explainability", short: "Explain", isGate: false, icon: "☀" },
  { key: "deploy", label: "Deployment", short: "Deploy", isGate: false, icon: "▲" },
  { key: "gate_deploy", label: "Gate 3 · Sign-off", short: "Gate 3", isGate: true, icon: "⛴" },
  { key: "report", label: "Report Generator", short: "Report", isGate: false, icon: "▦" },
];

export const STAGE_BY_KEY: Record<string, StageMeta> = Object.fromEntries(
  STAGES.map((s) => [s.key, s]),
);
