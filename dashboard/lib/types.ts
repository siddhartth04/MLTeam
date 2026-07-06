// Shared types mirroring the FastAPI response shapes.

export type StageStatus =
  | "pending"
  | "active"
  | "done"
  | "awaiting_approval"
  | "rejected"
  | "skipped"
  | "failed";

export type RunStatus =
  | "queued"
  | "running"
  | "awaiting_approval"
  | "completed"
  | "rejected"
  | "failed";

export interface StageState {
  stage: string;
  label: string;
  status: StageStatus;
  is_gate: boolean;
  detail: string;
  started_at: string | null;
  ended_at: string | null;
}

export interface RunSummary {
  ui_run_id: string;
  goal_text: string;
  dataset_name: string;
  status: RunStatus;
  created_at: string;
  result_run_id: string | null;
  final_report_id: string | null;
  llm_cost_usd: number;
  error: string | null;
  stages: StageState[];
}

export interface LogEvent {
  timestamp: string;
  level: "info" | "success" | "warning" | "error" | "gate";
  stage: string | null;
  message: string;
  extra: Record<string, unknown>;
}

export interface MetricSummary {
  mean: number;
  std: number;
}

export type ExperimentState = "completed" | "failed" | "skipped";

export interface ExperimentRow {
  artifact_id: string;
  experiment_id: string;
  template_id: string;
  config: Record<string, unknown>;
  state: ExperimentState;
  metrics: Record<string, MetricSummary>;
  wall_time_seconds: number | null;
  error: string | null;
  is_baseline: boolean;
}

export interface LineageNode {
  id: string;
  short_id: string;
  type: string;
  producer: string;
  protected: boolean;
}
export interface LineageEdge {
  source: string;
  target: string;
}
export interface LineageGraph {
  nodes: LineageNode[];
  edges: LineageEdge[];
}

export interface ArtifactDetail {
  id: string;
  short_id: string;
  type: string;
  producer: string;
  run_id: string;
  config_hash: string;
  created_at: string;
  parents: string[];
  children: string[];
  protected: boolean;
  payload: Record<string, unknown>;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
  std: number;
}
export interface FigureRef {
  kind: string;
  title: string;
  blob_hash: string;
  feature: string | null;
}
export interface Explainability {
  importances: FeatureImportance[];
  shap_available: boolean;
  figures: FigureRef[];
}

export interface Critique {
  verdict: "approve" | "iterate_with_actions" | "halt";
  findings: {
    severity: string;
    category: string;
    evidence: string;
    recommendation: string;
  }[];
  iteration_actions: string[];
}

export interface ProblemSpec {
  task_type: string;
  target_column: string;
  primary_metric: string;
  secondary_metrics: string[];
  business_constraints: string[];
  cost_asymmetry: { false_positive_cost: number; false_negative_cost: number; rationale: string };
  assumptions: string[];
  human_confirmed: boolean;
}

export interface DecisionRecord {
  decision: string;
  why: string;
  evidence: string[];
  alternatives_considered: string[];
  assumptions: string[];
}

export interface ExcludedTemplate {
  template_id: string;
  reason: string;
}

export interface DossierComparisonRow {
  experiment_id: string;
  template_id: string;
  state: string;
  primary_metric: string;
  mean: number | null;
  std: number | null;
  is_winner: boolean;
}

export interface DossierWinnerMargin {
  vs_runner_up: { template_id: string; improvement: number } | null;
  vs_dummy: number | null;
}

export interface DossierTopFeature {
  feature: string;
  importance: number;
}

export interface DossierCritique {
  verdict: string;
  findings: { severity: string; category: string; evidence: string }[];
}

export interface DeploymentDossier {
  comparison: DossierComparisonRow[];
  winner_margin: DossierWinnerMargin | null;
  top_features: DossierTopFeature[];
  critique: DossierCritique | null;
  lineage: { model_id: string; dataset_id: string; split_plan_id: string | null };
}

export interface PendingGate {
  ui_run_id: string;
  stage: string;
  summary: Record<string, unknown>;
}
