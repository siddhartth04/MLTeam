"use client";

// Renders a stored Plotly figure (fetched as JSON) themed for the light shell.
// Plotly is client-only, so it's dynamically imported with SSR disabled.

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";

import { figureUrl } from "@/lib/api";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const LIGHT_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#586173", family: "Inter, sans-serif", size: 12 },
  colorway: ["#4f46e5", "#0891b2", "#059669", "#d97706", "#dc2626"],
  margin: { l: 60, r: 20, t: 40, b: 50 },
  xaxis: { gridcolor: "#eceef2", zerolinecolor: "#dfe2e8" },
  yaxis: { gridcolor: "#eceef2", zerolinecolor: "#dfe2e8" },
};

export function PlotlyFigure({ blobHash, height = 320 }: { blobHash: string; height?: number }) {
  const [fig, setFig] = useState<{ data: unknown[]; layout: Record<string, unknown> } | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    fetch(figureUrl(blobHash))
      .then((r) => r.json())
      .then((j) => alive && setFig(j))
      .catch((e) => alive && setErr(String(e)));
    return () => {
      alive = false;
    };
  }, [blobHash]);

  if (err) return <div className="p-4 text-sm text-danger">Figure failed to load.</div>;
  if (!fig) return <div className="animate-pulse rounded-xl bg-base-700/40" style={{ height }} />;

  return (
    <Plot
      data={fig.data as never}
      layout={{ ...LIGHT_LAYOUT, ...(fig.layout as object), autosize: true, height } as never}
      config={{ displayModeBar: false, responsive: true } as never}
      style={{ width: "100%", height }}
      useResizeHandler
    />
  );
}
