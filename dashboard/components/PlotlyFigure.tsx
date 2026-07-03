"use client";

// Renders a stored Plotly figure (fetched as JSON) themed for the dark shell.
// Plotly is client-only, so it's dynamically imported with SSR disabled.

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";

import { figureUrl } from "@/lib/api";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const DARK_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#9aa6c8", family: "Inter, sans-serif", size: 12 },
  colorway: ["#7c5cff", "#31d0ff", "#38e8a0", "#ffb648", "#ff5d73"],
  margin: { l: 60, r: 20, t: 40, b: 50 },
  xaxis: { gridcolor: "#1c2440", zerolinecolor: "#28324f" },
  yaxis: { gridcolor: "#1c2440", zerolinecolor: "#28324f" },
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
      layout={{ ...DARK_LAYOUT, ...(fig.layout as object), autosize: true, height } as never}
      config={{ displayModeBar: false, responsive: true } as never}
      style={{ width: "100%", height }}
      useResizeHandler
    />
  );
}
