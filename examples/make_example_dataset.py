"""Generate the deterministic example dataset: customer churn (binary).

Deliberately includes the messiness the pipeline must handle:
- customer_id: unique identifier (should be dropped)
- account_status: constant column (should be dropped)
- textual missing tokens ("N/A", "") in categorical columns
- real nulls in numeric columns
- a few exact duplicate rows

Run:  uv run python examples/make_example_dataset.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

SEED = 7
N_ROWS = 600
OUTPUT = Path(__file__).parent / "customer_churn.csv"


def make_churn_frame(n_rows: int = N_ROWS, seed: int = SEED) -> pl.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 80, n_rows).astype(float)
    tenure_months = rng.integers(1, 72, n_rows).astype(float)
    monthly_charges = np.round(rng.uniform(20, 120, n_rows), 2)
    support_calls = rng.poisson(1.5, n_rows).astype(float)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], n_rows, p=[0.5, 0.3, 0.2])
    payment = rng.choice(
        ["card", "bank-transfer", "e-check", "N/A"], n_rows, p=[0.4, 0.3, 0.25, 0.05]
    )

    # Churn probability: short tenure, month-to-month, many support calls.
    logit = (
        -1.2
        - 0.03 * tenure_months
        + 0.55 * support_calls
        + 0.012 * monthly_charges
        + np.where(contract == "month-to-month", 1.1, 0.0)
        - 0.01 * age
    )
    churned = (rng.uniform(size=n_rows) < 1 / (1 + np.exp(-logit))).astype(int)

    # Inject numeric nulls (~4% of ages, ~3% of charges).
    age[rng.uniform(size=n_rows) < 0.04] = np.nan
    monthly_charges[rng.uniform(size=n_rows) < 0.03] = np.nan

    df = pl.DataFrame(
        {
            "customer_id": [f"CUST-{i:05d}" for i in range(n_rows)],
            "age": age,
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "num_support_calls": support_calls,
            "contract_type": contract,
            "payment_method": payment,
            "account_status": ["active"] * n_rows,
            "churned": churned,
        }
    )
    # Append a few exact duplicates (ids included) to exercise dedup.
    return pl.concat([df, df.head(5)])


if __name__ == "__main__":
    frame = make_churn_frame()
    frame.write_csv(OUTPUT)
    print(f"wrote {frame.height} rows x {frame.width} cols -> {OUTPUT}")
