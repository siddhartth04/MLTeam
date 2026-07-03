"use client";

// Subscribe to a run's SSE stream and maintain live stage + log + status state.
// Falls back gracefully: if the stream closes, the caller's SWR polling keeps
// things fresh. Reconnects when the run id changes.

import { useEffect, useRef, useState } from "react";

import type { LogEvent, RunStatus, StageState } from "./types";

export interface RunStreamState {
  stages: StageState[];
  events: LogEvent[];
  status: RunStatus | null;
  cost: number;
  gate: string | null;
  connected: boolean;
}

const EMPTY: RunStreamState = {
  stages: [],
  events: [],
  status: null,
  cost: 0,
  gate: null,
  connected: false,
};

export function useRunStream(uiRunId: string | null): RunStreamState {
  const [state, setState] = useState<RunStreamState>(EMPTY);
  const stagesRef = useRef<Map<string, StageState>>(new Map());

  useEffect(() => {
    if (!uiRunId) {
      setState(EMPTY);
      return;
    }
    stagesRef.current = new Map();
    setState({ ...EMPTY });

    const es = new EventSource(`/api/runs/${uiRunId}/stream`);

    const commitStages = () =>
      setState((s) => ({ ...s, stages: Array.from(stagesRef.current.values()) }));

    es.addEventListener("open", () => setState((s) => ({ ...s, connected: true })));

    es.addEventListener("snapshot", (e) => {
      const data = JSON.parse((e as MessageEvent).data);
      stagesRef.current = new Map(
        (data.stages as StageState[]).map((st) => [st.stage, st]),
      );
      setState({
        stages: Array.from(stagesRef.current.values()),
        events: data.events ?? [],
        status: data.status,
        cost: data.llm_cost_usd ?? 0,
        gate: data.gate ?? null,
        connected: true,
      });
    });

    es.addEventListener("stage", (e) => {
      const st = JSON.parse((e as MessageEvent).data) as StageState;
      stagesRef.current.set(st.stage, st);
      commitStages();
    });

    es.addEventListener("log", (e) => {
      const ev = JSON.parse((e as MessageEvent).data) as LogEvent;
      setState((s) => ({ ...s, events: [...s.events.slice(-400), ev] }));
    });

    es.addEventListener("status", (e) => {
      const data = JSON.parse((e as MessageEvent).data);
      setState((s) => ({ ...s, status: data.status, gate: data.gate ?? s.gate }));
    });

    es.addEventListener("done", () => {
      es.close();
      setState((s) => ({ ...s, connected: false }));
    });

    es.onerror = () => {
      setState((s) => ({ ...s, connected: false }));
      es.close();
    };

    return () => es.close();
  }, [uiRunId]);

  return state;
}
