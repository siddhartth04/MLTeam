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

const MAX_RECONNECTS = 5;
const RECONNECT_DELAY_MS = 2500;

// Defensive JSON.parse: a malformed SSE frame should skip that frame, not
// throw an unhandled error inside the listener.
function parse<T>(e: Event): T | null {
  try {
    return JSON.parse((e as MessageEvent).data) as T;
  } catch {
    return null;
  }
}

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

    let es: EventSource | null = null;
    let reconnects = 0;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let done = false;

    const commitStages = () =>
      setState((s) => ({ ...s, stages: Array.from(stagesRef.current.values()) }));

    const connect = () => {
      es = new EventSource(`/api/runs/${encodeURIComponent(uiRunId)}/stream`);

      es.addEventListener("open", () => {
        reconnects = 0; // a healthy connection resets the retry budget
        setState((s) => ({ ...s, connected: true }));
      });

      es.addEventListener("snapshot", (e) => {
        const data = parse<Record<string, unknown>>(e);
        if (!data) return;
        stagesRef.current = new Map(
          (data.stages as StageState[]).map((st) => [st.stage, st]),
        );
        setState({
          stages: Array.from(stagesRef.current.values()),
          events: (data.events as LogEvent[]) ?? [],
          status: data.status as RunStreamState["status"],
          cost: (data.llm_cost_usd as number) ?? 0,
          gate: (data.gate as string) ?? null,
          connected: true,
        });
      });

      es.addEventListener("stage", (e) => {
        const st = parse<StageState>(e);
        if (!st) return;
        stagesRef.current.set(st.stage, st);
        commitStages();
      });

      es.addEventListener("log", (e) => {
        const ev = parse<LogEvent>(e);
        if (!ev) return;
        setState((s) => ({ ...s, events: [...s.events.slice(-400), ev] }));
      });

      es.addEventListener("status", (e) => {
        const data = parse<Record<string, unknown>>(e);
        if (!data) return;
        setState((s) => ({
          ...s,
          status: data.status as RunStreamState["status"],
          gate: (data.gate as string) ?? s.gate,
        }));
      });

      es.addEventListener("done", () => {
        done = true;
        es?.close();
        setState((s) => ({ ...s, connected: false }));
      });

      es.onerror = () => {
        setState((s) => ({ ...s, connected: false }));
        es?.close();
        // Try a bounded number of manual reconnects before giving up and
        // letting the caller's SWR polling take over.
        if (!done && reconnects < MAX_RECONNECTS) {
          reconnects += 1;
          reconnectTimer = setTimeout(connect, RECONNECT_DELAY_MS);
        }
      };
    };

    connect();

    return () => {
      done = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      es?.close();
    };
  }, [uiRunId]);

  return state;
}
