"use client";

// Selected-run context, persisted to the URL (?run=) so deep links work and
// every page shares the same active run.

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { createContext, useContext, useMemo, type ReactNode } from "react";

interface RunCtx {
  runId: string | null;
  setRunId: (id: string) => void;
}

const Ctx = createContext<RunCtx>({ runId: null, setRunId: () => {} });

export function RunProvider({ children }: { children: ReactNode }) {
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const runId = params.get("run");

  const value = useMemo<RunCtx>(
    () => ({
      runId,
      setRunId: (id: string) => {
        // URLSearchParams encodes values; guards against malformed ids in the URL.
        const next = new URLSearchParams(params.toString());
        next.set("run", id);
        router.push(`${pathname}?${next.toString()}`);
      },
    }),
    [runId, params, pathname, router],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useActiveRun() {
  return useContext(Ctx);
}
