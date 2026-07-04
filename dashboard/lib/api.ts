// API client + SWR fetcher. All requests hit same-origin /api/* (proxied to
// FastAPI in dev via next.config rewrites), so SSE and CORS just work.

import useSWR from "swr";

import type {
  ArtifactDetail,
  Critique,
  Explainability,
  ExperimentRow,
  LineageGraph,
  LogEvent,
  PendingGate,
  ProblemSpec,
  RunSummary,
} from "./types";

export const fetcher = async (url: string) => {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
};

// User-controlled ids (from ?run=) go into URL paths — always encode them.
const enc = encodeURIComponent;

export function useRuns() {
  return useSWR<RunSummary[]>("/api/runs", fetcher, { refreshInterval: 4000 });
}

export function useRun(uiRunId: string | null, refresh = 0) {
  return useSWR<RunSummary>(uiRunId ? `/api/runs/${enc(uiRunId)}` : null, fetcher, {
    refreshInterval: refresh,
  });
}

export function useRunEvents(uiRunId: string | null, refresh = 0) {
  return useSWR<LogEvent[]>(uiRunId ? `/api/runs/${enc(uiRunId)}/events` : null, fetcher, {
    refreshInterval: refresh,
  });
}

export function useExperiments(uiRunId: string | null) {
  return useSWR<ExperimentRow[]>(uiRunId ? `/api/runs/${enc(uiRunId)}/experiments` : null, fetcher);
}

export function useLineage(uiRunId: string | null) {
  return useSWR<LineageGraph>(uiRunId ? `/api/runs/${enc(uiRunId)}/lineage` : null, fetcher);
}

export function useArtifact(artifactId: string | null) {
  return useSWR<ArtifactDetail>(artifactId ? `/api/artifacts/${enc(artifactId)}` : null, fetcher);
}

export function useExplainability(uiRunId: string | null) {
  return useSWR<Explainability | null>(
    uiRunId ? `/api/runs/${enc(uiRunId)}/explainability` : null,
    fetcher,
  );
}

export function useCritique(uiRunId: string | null) {
  return useSWR<Critique | null>(uiRunId ? `/api/runs/${enc(uiRunId)}/critique` : null, fetcher);
}

export function useProblemSpec(uiRunId: string | null) {
  return useSWR<ProblemSpec | null>(uiRunId ? `/api/runs/${enc(uiRunId)}/problem-spec` : null, fetcher);
}

export function useGates() {
  return useSWR<PendingGate[]>("/api/gates", fetcher, { refreshInterval: 2000 });
}

export async function launchRun(opts: {
  goalText: string;
  file: File | null;
  demo: boolean;
}): Promise<string> {
  const form = new FormData();
  form.append("goal_text", opts.goalText);
  form.append("demo", String(opts.demo));
  if (opts.file) form.append("file", opts.file);
  const res = await fetch("/api/runs", { method: "POST", body: form });
  if (!res.ok) throw new Error(`launch failed: ${res.status}`);
  const data = await res.json();
  return data.ui_run_id as string;
}

export async function resolveGate(
  uiRunId: string,
  approved: boolean,
  notes = "",
): Promise<void> {
  const res = await fetch(`/api/runs/${enc(uiRunId)}/gate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ approved, notes }),
  });
  if (!res.ok) throw new Error(`gate action failed: ${res.status}`);
}

export function figureUrl(blobHash: string): string {
  return `/api/figures/${enc(blobHash)}`;
}

export function reportUrl(reportId: string, kind: "business" | "technical"): string {
  return `/api/reports/${enc(reportId)}/${kind}`;
}
