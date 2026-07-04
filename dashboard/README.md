# Mission Control — the dashboard

A bespoke control-room UI for the AI ML Engineering Team: watch the agents
profile, train, critique, and ship — live — approve at the three human gates,
and read the final reports. Next.js 14 (app router) + TypeScript + Tailwind +
Framer Motion + Plotly, over the FastAPI read layer in `src/mlteam/api/`.

## Run it

Two processes: the API (Python) and the dashboard (Node).

```bash
# 1. API — from the repo root. Seeds a completed demo run on first start so
#    every page is populated immediately (no API key needed).
uv run uvicorn mlteam.api.app:app --port 8000

# 2. Dashboard — from dashboard/
cd dashboard
npm install          # first time only
npm run dev          # http://localhost:3000
```

The dashboard proxies `/api/*` to `http://localhost:8000` (see
`next.config.mjs`), so the browser talks to a single origin and SSE streams
pass straight through. Point elsewhere with `MLTEAM_API_URL`.

Live runs use Claude if `ANTHROPIC_API_KEY` is set; otherwise every run
(including the "Live" button) falls back to the recorded fixture, so the whole
experience works with no credentials.

## The pages

- **Launch** — upload a CSV (or let it synthesize one), describe your goal, go.
- **Mission Control** — the animated hub-and-spoke agent flow, the streaming
  activity feed, live cost/budget meters, and the stage rail.
- **Approval Gates** — the three human checkpoints as review cards with
  Approve / Reject.
- **Experiments** — the model leaderboard with CV-metric bars, best highlighted.
- **Lineage** — the artifact provenance DAG; click any node to inspect it.
- **Trace** — the full per-run audit timeline, filterable by level.
- **Report** — the rendered business & technical HTML reports and the
  explainability figures, inline.

## Build

```bash
npm run build    # production build + type check
```
