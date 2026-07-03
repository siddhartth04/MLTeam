"""Reference-score comparison: the pass/fail logic of the regression suite."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REFERENCE_PATH = Path(__file__).parent / "reference_scores.json"

# |current - reference| must stay within 2% of the reference magnitude
# (floored for metrics near zero). Generous enough for cross-machine float
# noise, tight enough to catch real regressions.
_RELATIVE_TOLERANCE = 0.02
_ABSOLUTE_FLOOR = 0.005


@dataclass(frozen=True)
class ComparisonRow:
    name: str
    metric: str
    reference: float | None
    current: float
    delta: float | None
    status: str  # "PASS" | "FAIL" | "NEW"


def tolerance_for(reference: float) -> float:
    return max(_RELATIVE_TOLERANCE * abs(reference), _ABSOLUTE_FLOOR)


def compare_scores(
    current: dict[str, dict[str, float | str]],
    reference: dict[str, dict[str, float | str]],
) -> tuple[ComparisonRow, ...]:
    """Compare current benchmark scores against stored references."""
    rows = []
    for name, entry in current.items():
        metric = str(entry["metric"])
        mean = float(entry["mean"])
        ref_entry = reference.get(name)
        if ref_entry is None:
            rows.append(
                ComparisonRow(
                    name=name, metric=metric, reference=None, current=mean, delta=None, status="NEW"
                )
            )
            continue
        ref_mean = float(ref_entry["mean"])
        delta = mean - ref_mean
        status = "PASS" if abs(delta) <= tolerance_for(ref_mean) else "FAIL"
        rows.append(
            ComparisonRow(
                name=name,
                metric=metric,
                reference=ref_mean,
                current=mean,
                delta=round(delta, 6),
                status=status,
            )
        )
    return tuple(rows)


def load_references(path: Path = REFERENCE_PATH) -> dict[str, dict[str, float | str]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_references(scores: dict[str, dict[str, float | str]], path: Path = REFERENCE_PATH) -> None:
    path.write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_table(rows: tuple[ComparisonRow, ...]) -> str:
    header = (
        f"{'dataset':<16} {'metric':<10} {'reference':>10} {'current':>10} {'delta':>10}  status"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        ref = f"{row.reference:.4f}" if row.reference is not None else "-"
        delta = f"{row.delta:+.4f}" if row.delta is not None else "-"
        lines.append(
            f"{row.name:<16} {row.metric:<10} {ref:>10} "
            f"{row.current:>10.4f} {delta:>10}  {row.status}"
        )
    return "\n".join(lines)
