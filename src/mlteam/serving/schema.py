"""Inference schema: a declarative column contract derived from the profile.

The SAME schema enforced here validates requests at serving time — that is
what kills train/serve skew. It is stored as a small JSON spec (portable, no
pickled Python) and materialized into a Pandera DataFrameSchema on demand.
"""

from __future__ import annotations

import json
from typing import Any

from mlteam.contracts import DataProfile

# Polars dtype string -> ("pandas_dtype", "kind") for the request contract.
_NUMERIC_DTYPES = ("Int", "UInt", "Float")


def _column_kind(dtype: str) -> str:
    if any(dtype.startswith(prefix) for prefix in _NUMERIC_DTYPES):
        return "numeric"
    if dtype in ("Date", "Datetime"):
        return "datetime"
    return "categorical"


def build_inference_schema_spec(
    profile: DataProfile, *, target_column: str, feature_names: list[str]
) -> dict[str, Any]:
    """JSON-able inference schema spec for the served features (no target)."""
    columns: list[dict[str, Any]] = []
    by_name = {c.name: c for c in profile.columns}
    for name in feature_names:
        if name == target_column or name not in by_name:
            continue
        col = by_name[name]
        columns.append(
            {
                "name": name,
                "kind": _column_kind(col.dtype),
                # Columns that had no missing values in training are required
                # non-null at inference; others may be null.
                "nullable": col.missing_pct > 0.0,
            }
        )
    return {"version": 1, "columns": columns}


def spec_to_pandera(spec: dict[str, Any]) -> Any:
    """Materialize a Pandera DataFrameSchema from a stored spec."""
    import pandera.pandas as pa

    kind_to_check = {
        "numeric": pa.Column(float, nullable=True, coerce=True, required=True),
        "categorical": pa.Column(object, nullable=True, coerce=True, required=True),
        "datetime": pa.Column(object, nullable=True, coerce=True, required=True),
    }
    columns = {}
    for entry in spec["columns"]:
        base = kind_to_check[entry["kind"]]
        columns[entry["name"]] = pa.Column(
            base.dtype,
            nullable=entry["nullable"],
            coerce=True,
            required=True,
        )
    return pa.DataFrameSchema(columns, strict=False)


def serialize_spec(spec: dict[str, Any]) -> bytes:
    return json.dumps(spec, indent=2, sort_keys=True).encode("utf-8")


def load_spec(raw: bytes) -> dict[str, Any]:
    return json.loads(raw.decode("utf-8"))
