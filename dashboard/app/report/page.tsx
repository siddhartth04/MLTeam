"use client";

import { useState } from "react";

import { PlotlyFigure } from "@/components/PlotlyFigure";
import { useActiveRun } from "@/components/RunContext";
import { EmptyState, Panel, SectionTitle, StatTile, VerdictBadge } from "@/components/primitives";
import {
  checkDrift,
  reportDownloadUrl,
  reportUrl,
  useCritique,
  useExplainability,
  useRun,
} from "@/lib/api";
import type { DriftSummary } from "@/lib/types";

export default function ReportPage() {
  const { runId } = useActiveRun();
  const { data: run } = useRun(runId, 4000);
  const { data: explain } = useExplainability(runId);
  const { data: critique } = useCritique(runId);
  const [tab, setTab] = useState<"business" | "technical">("business");

  if (!runId) {
    return (
      <Panel>
        <EmptyState icon="▦" title="No run selected" hint="Pick a completed run to view its reports." />
      </Panel>
    );
  }
  const reportId = run?.final_report_id;

  return (
    <div>
      <SectionTitle sub="The deliverables: a business narrative and a technical deep-dive, plus explainability. Holdout metrics are computed here — the only time the frozen set is read.">
        Reports
      </SectionTitle>

      {!reportId ? (
        <Panel>
          <EmptyState
            icon="▦"
            title="No report yet"
            hint="The report is generated at the end of a run, after the Critic and deployment."
          />
        </Panel>
      ) : (
        <div className="space-y-6">
          {critique && (
            <div className="flex flex-wrap items-center gap-4">
              <StatTile label="Critic verdict" value={<VerdictBadge verdict={critique.verdict} />} />
              {explain && (
                <StatTile
                  label="Top driver"
                  value={<span className="text-lg">{explain.importances[0]?.feature ?? "—"}</span>}
                  sub="by permutation importance"
                  accent="cyan"
                />
              )}
              {explain && (
                <StatTile
                  label="SHAP"
                  value={<span className="text-lg">{explain.shap_available ? "available" : "n/a"}</span>}
                  accent={explain.shap_available ? "ok" : "warn"}
                />
              )}
            </div>
          )}

          {explain && explain.figures.length > 0 && (
            <Panel eyebrow="Explainability" title="What drives the model">
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                {explain.figures.map((f) => (
                  <div key={f.blob_hash} className="glass-raised p-3">
                    <div className="mb-2 text-sm font-semibold text-ink">{f.title}</div>
                    <PlotlyFigure blobHash={f.blob_hash} height={300} />
                  </div>
                ))}
              </div>
            </Panel>
          )}

          <DriftCard runId={runId} />

          <Panel
            eyebrow="Deliverable"
            title="Generated report"
            action={
              <div className="flex gap-1.5 rounded-xl bg-base-700/50 p-1">
                {(["business", "technical"] as const).map((t) => (
                  <button
                    key={t}
                    onClick={() => setTab(t)}
                    className={`rounded-lg px-4 py-1.5 text-sm font-semibold capitalize transition-colors ${
                      tab === t ? "bg-accent/20 text-accent-bright" : "text-ink-muted hover:text-ink"
                    }`}
                  >
                    {t}
                  </button>
                ))}
              </div>
            }
          >
            <iframe
              key={tab}
              src={reportUrl(reportId, tab)}
              title={`${tab} report`}
              // The report is our own Jinja2-autoescaped HTML, but sandbox it
              // as defense-in-depth: scripts (Plotly) and same-origin asset
              // loads are allowed; forms, popups, and top-nav are not.
              sandbox="allow-scripts allow-same-origin"
              className="report-frame h-[720px] w-full border-0"
            />
            <div className="mt-3 flex justify-end gap-3">
              <a href={reportDownloadUrl(reportId, tab)} download className="btn-ghost text-xs">
                ⬇ Download
              </a>
              <a
                href={reportUrl(reportId, tab)}
                target="_blank"
                rel="noreferrer"
                className="btn-ghost text-xs"
              >
                Open in new tab ↗
              </a>
            </div>
          </Panel>
        </div>
      )}
    </div>
  );
}

function DriftCard({ runId }: { runId: string }) {
  const [summary, setSummary] = useState<DriftSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function handleFile(file: File) {
    setBusy(true);
    setError(null);
    try {
      setSummary(await checkDrift(runId, file));
    } catch (err) {
      setError(err instanceof Error ? err.message : "drift check failed");
      setSummary(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Panel
      eyebrow="Monitoring"
      title="Data drift"
      action={
        <label className="btn-ghost cursor-pointer text-xs">
          {busy ? "Checking…" : "Upload current data (.csv)"}
          <input
            type="file"
            accept=".csv,.parquet"
            className="hidden"
            disabled={busy}
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) void handleFile(file);
              e.target.value = "";
            }}
          />
        </label>
      }
    >
      {error && <div className="mb-3 text-sm text-danger">{error}</div>}
      {!summary ? (
        <EmptyState
          icon="◈"
          title="No drift check run yet"
          hint="Upload a CSV of recent production data to compare it against this model's training baseline (Evidently's data-drift preset)."
        />
      ) : (
        <div className="space-y-4">
          <div className="flex flex-wrap gap-4">
            <StatTile
              label="Drifted columns"
              value={`${summary.n_drifted_columns} / ${summary.n_columns_checked}`}
              accent={summary.n_drifted_columns > 0 ? "warn" : "ok"}
            />
            <StatTile
              label="Share drifted"
              value={`${(summary.share_drifted * 100).toFixed(1)}%`}
              accent={summary.share_drifted > 0 ? "warn" : "ok"}
            />
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-ink-faint">
                <th className="pb-2 font-medium">Column</th>
                <th className="pb-2 font-medium">Drift score</th>
                <th className="pb-2 font-medium">Drifted</th>
              </tr>
            </thead>
            <tbody>
              {summary.columns.map((c) => (
                <tr key={c.column} className="border-t border-base-700/50">
                  <td className="py-1.5">{c.column}</td>
                  <td className="py-1.5">{c.drift_score?.toFixed(4) ?? "—"}</td>
                  <td className={`py-1.5 ${c.drifted ? "text-warn" : "text-ok"}`}>
                    {c.drifted ? "yes" : "no"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}
