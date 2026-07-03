"""Deterministic synthetic datasets shared by test modules."""

from __future__ import annotations

import numpy as np
import polars as pl


def make_churn_frame(n_rows: int = 240, seed: int = 11, leaky: bool = False) -> pl.DataFrame:
    """Small deterministic churn-like frame for tests.

    Includes an ID column, a constant column, textual missing tokens, numeric
    nulls, and duplicates. ``leaky=True`` adds a feature that is the target
    plus tiny noise — the leakage checks must catch it.
    """
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n_rows).astype(float)
    calls = rng.poisson(1.5, n_rows).astype(float)
    charges = np.round(rng.uniform(20, 120, n_rows), 2)
    contract = rng.choice(["monthly", "yearly", "N/A"], n_rows, p=[0.55, 0.4, 0.05])
    logit = -1.0 - 0.03 * tenure + 0.6 * calls + np.where(contract == "monthly", 1.0, 0.0)
    target = (rng.uniform(size=n_rows) < 1 / (1 + np.exp(-logit))).astype(int)
    charges[rng.uniform(size=n_rows) < 0.05] = np.nan

    data = {
        "row_id": [f"R-{i:04d}" for i in range(n_rows)],
        "tenure_months": tenure,
        "num_support_calls": calls,
        "monthly_charges": charges,
        "contract_type": contract,
        "region": ["north"] * n_rows,
        "churned": target,
    }
    if leaky:
        data["leaked_score"] = target + rng.normal(0.0, 0.01, n_rows)
    df = pl.DataFrame(data)
    return pl.concat([df, df.head(3)])
