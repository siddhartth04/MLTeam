"""Column preview: cheap per-column stats available BEFORE Gate 1.

The requirement agent used to see column NAMES ONLY (``_read_columns`` in
``orchestration/agentic.py``), so it could not rank candidate targets or
warn about a likely duplicate/leaky column — it was choosing blind. This
module gives it null rates, cardinality, an inferred type, and a few sample
values instead, computed deterministically (no LLM, no artifact) on a capped
row sample so it stays cheap on large files.

Reads with ``infer_schema=False`` (like ``_read_columns``) so a column with
a stray non-numeric value (a Kaggle "> 10" bucket label, say) can never
crash the preview — every column is text until THIS module classifies it,
and classification itself never raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from mlteam.agents.triage import _numeric_parse_expr, _present_mask

_DATE_PARSE_RATE_MIN = 0.8
_NUMERIC_PARSE_RATE_MIN = 0.8
_MAX_SAMPLE_VALUES = 3


@dataclass(frozen=True)
class ColumnPreview:
    name: str
    inferred_type: str  # "numeric" | "date" | "text" | "empty"
    null_fraction: float
    n_unique: int
    sample_values: tuple[str, ...]


def _classify(df: pl.DataFrame, column: str, present: pl.Series) -> str:
    n_present = int(present.sum())
    if n_present == 0:
        return "empty"

    # Date must be checked BEFORE numeric: the shared numeric-parse regex
    # (also used by triage's currency/unit coercion, which needs leading
    # partial matches like "45.5 Lac") happily reads a leading "2024" out of
    # "2024-01-15" as the number 2024 — so a date-like column would otherwise
    # misclassify as numeric.
    #
    # to_date raises ComputeError (not just per-value nulls) when polars can't
    # infer ANY date format across the whole column — exactly the ordinary
    # case of a genuinely non-date text column, which this must not crash on.
    try:
        dates = df.select(pl.col(column).str.to_date(strict=False).alias("d"))["d"]
        date_rate = int((dates.is_not_null() & present).sum()) / n_present
    except pl.exceptions.ComputeError:
        date_rate = 0.0
    if date_rate >= _DATE_PARSE_RATE_MIN:
        return "date"

    numeric = df.select(_numeric_parse_expr(column).alias("v"))["v"]
    numeric_rate = int((numeric.is_not_null() & present).sum()) / n_present
    if numeric_rate >= _NUMERIC_PARSE_RATE_MIN:
        return "numeric"

    return "text"


def column_preview(data_path: Path, n_rows: int = 5000) -> list[ColumnPreview]:
    """Deterministic per-column stats on a capped sample. Never raises on junk."""
    if data_path.suffix.lower() == ".parquet":
        df = pl.read_parquet(data_path, n_rows=n_rows)
        df = df.select(pl.all().cast(pl.String))
    else:
        df = pl.read_csv(data_path, n_rows=n_rows, infer_schema=False)

    previews: list[ColumnPreview] = []
    for column in df.columns:
        present = _present_mask(df, column)
        n_present = int(present.sum())
        null_fraction = 1.0 - (n_present / len(df) if len(df) else 0.0)
        n_unique = df.filter(present)[column].n_unique() if n_present else 0
        samples = (
            df.filter(present)[column].unique(maintain_order=True).head(_MAX_SAMPLE_VALUES)
            if n_present
            else df[column].head(0)
        )
        previews.append(
            ColumnPreview(
                name=column,
                inferred_type=_classify(df, column, present),
                null_fraction=round(null_fraction, 4),
                n_unique=n_unique,
                sample_values=tuple(str(v) for v in samples.to_list()),
            )
        )
    return previews
