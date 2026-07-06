"""Data triage: senior-engineer readiness decisions before the pipeline runs.

Every uploaded dataset passes through here after Gate 1 (so the target and
task type are confirmed) and before profiling/splitting. Deterministic
detectors find trainability blockers; vetted remedies fix the fixable ones;
thresholds encode the judgment a careful engineer would apply; and the whole
outcome is stored as a TriageReport artifact and surfaced at Gate 2.

The remedy catalog (all row-local, all logged):
- drop_unlabeled_rows     — rows with no target value are useless for
                            supervised learning; drop them, loudly.
- coerce_target_numeric   — a regression/forecasting target stored as text
                            ("₹45,00,000", "45.5 Lac", "1,200") is parsed with
                            currency/thousands stripping and a vetted unit-
                            multiplier table. Applied only when >=95% of
                            present values parse; otherwise the run halts
                            with an actionable message (the column is
                            probably not a usable target).
- coerce_column_numeric   — a feature column that is >=80% numeric but
                            polluted with junk ("> 10", "1,200") becomes
                            numeric with junk as null (the in-pipeline
                            imputer handles the nulls). Genuinely
                            categorical text columns parse ~0% and are
                            left alone.
- drop_duplicate_rows     — exact-duplicate rows are dropped pre-split.
                            This is the ONE place duplicates may safely be
                            removed: doing it post-split (as the cleaner's
                            equivalent op does, for direct callers that skip
                            triage) risks an identical row landing in both
                            train and holdout, inflating holdout metrics with
                            memorization the split boundary can't detect.

Hard stops (clear ValueError, never a stack trace from deep inside sklearn):
- target text that mostly fails to parse for a regression task;
- fewer than MIN_LABELED_ROWS labeled rows after dropping unlabeled ones.

New real-world failure modes should become new detectors/remedies HERE —
one taxonomy, one catalog — not scattered patches in ingest/train code.
"""

from __future__ import annotations

import polars as pl
import structlog

from mlteam.agents.data_io import read_dataset, store_dataframe
from mlteam.contracts import (
    DatasetRole,
    TaskType,
    TriageAction,
    TriageIssue,
    TriageIssueCode,
    TriageRemedy,
    TriageReport,
    is_classification,
    new_artifact,
)
from mlteam.contracts.dataset import DatasetVersion
from mlteam.store.interface import ArtifactStore

_log = structlog.get_logger(__name__)

MIN_LABELED_ROWS = 50
TARGET_PARSE_RATE_MIN = 0.95
FEATURE_PARSE_RATE_MIN = 0.80
WARN_DROP_FRACTION = 0.40

# Same family of missing-value spellings the cleaner normalizes post-split.
_MISSING_TOKENS = ("", "na", "n/a", "null", "none", "nan", "?", "-", "missing")

# Vetted magnitude words seen in real price/quantity columns. The LLM never
# extends this table; unknown units multiply by 1 (the number still parses).
_UNIT_MULTIPLIERS: dict[str, float] = {
    "k": 1e3,
    "thousand": 1e3,
    "lakh": 1e5,
    "lac": 1e5,
    "m": 1e6,
    "mn": 1e6,
    "million": 1e6,
    "cr": 1e7,
    "crore": 1e7,
    "b": 1e9,
    "bn": 1e9,
    "billion": 1e9,
}


def _cleaned_text(column: str) -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.String)
        .str.strip_chars()
        .str.to_lowercase()
        .str.replace_all(r"[₹$€£,]", "")
        .str.strip_chars()
    )


def _numeric_parse_expr(column: str) -> pl.Expr:
    """Parse text to float: leading number x vetted unit multiplier, else null."""
    cleaned = _cleaned_text(column)
    number = cleaned.str.extract(r"^(-?\d*\.?\d+)", 1).cast(pl.Float64, strict=False)
    unit = cleaned.str.extract(r"^-?\d*\.?\d+\s*([a-z]+)", 1)
    multiplier = (
        pl.when(unit.is_null())
        .then(pl.lit(1.0))
        .otherwise(unit.replace_strict(_UNIT_MULTIPLIERS, default=1.0))
    )
    return number * multiplier


def _present_mask(df: pl.DataFrame, column: str) -> pl.Series:
    """True where the value is genuinely present (not null / sentinel token)."""
    if df.schema[column] == pl.String:
        cleaned = df.select(_cleaned_text(column).alias("c"))["c"]
        return cleaned.is_not_null() & ~cleaned.is_in(_MISSING_TOKENS)
    return df[column].is_not_null()


def _parse_rate(df: pl.DataFrame, column: str, present: pl.Series) -> tuple[float, int]:
    """(fraction of present values that parse as numbers, count that do not)."""
    n_present = int(present.sum())
    if n_present == 0:
        return 0.0, 0
    parsed = df.select(_numeric_parse_expr(column).alias("p"))["p"]
    n_parsed = int((parsed.is_not_null() & present).sum())
    return n_parsed / n_present, n_present - n_parsed


def triage_dataset(
    store: ArtifactStore,
    dataset_id: str,
    *,
    target_column: str,
    task_type: TaskType,
    run_id: str,
) -> tuple[DatasetVersion, TriageReport]:
    """Scan + remedy a raw dataset; returns (modeling dataset, report).

    The returned dataset equals the input when the data was already clean.
    Raises ValueError with an actionable message on unsalvageable data.
    """
    dataset = store.get(dataset_id)
    df = read_dataset(store, dataset)
    rows_before = len(df)
    issues: list[TriageIssue] = []
    remedies: list[TriageRemedy] = []
    changed = False

    # -- 1. Regression-like target stored as text -----------------------------
    regression_like = not is_classification(task_type)
    if regression_like and df.schema[target_column] == pl.String:
        present = _present_mask(df, target_column)
        rate, n_junk = _parse_rate(df, target_column, present)
        issues.append(
            TriageIssue(
                code=TriageIssueCode.TARGET_NOT_NUMERIC,
                column=target_column,
                detail=f"{task_type.value} target is text; {rate:.1%} of present values parse",
                affected_rows=int(present.sum()),
                fraction=round(1.0 - rate, 6),
            )
        )
        if rate < TARGET_PARSE_RATE_MIN:
            raise ValueError(
                f"target column '{target_column}' is text and only {rate:.1%} of its "
                f"values parse as numbers — it does not look like a usable "
                f"{task_type.value} target. Check whether another column holds the "
                "numeric value, or clean this column before uploading."
            )
        df = df.with_columns(_numeric_parse_expr(target_column).alias(target_column))
        changed = True
        remedies.append(
            TriageRemedy(
                action=TriageAction.COERCE_TARGET_NUMERIC,
                column=target_column,
                affected_rows=n_junk,
                justification=(
                    f"parsed text target to numbers ({rate:.1%} parse rate; currency/"
                    f"thousands separators stripped, vetted unit multipliers applied); "
                    f"{n_junk} unparseable value(s) became missing labels"
                ),
            )
        )

    # -- 2. Rows without a target label ---------------------------------------
    # For string targets only null/blank count as missing: a sentinel-looking
    # word ("none", "na") could be a legitimate class label — dropping those
    # rows would silently corrupt a classification dataset.
    if df.schema[target_column] == pl.String:
        stripped = df.select(pl.col(target_column).str.strip_chars().alias("c"))["c"]
        missing_mask = stripped.is_null() | (stripped == "")
    else:
        missing_mask = df[target_column].is_null()
    n_missing = int(missing_mask.sum())
    if n_missing > 0:
        fraction = n_missing / rows_before
        issues.append(
            TriageIssue(
                code=TriageIssueCode.TARGET_MISSING_ROWS,
                column=target_column,
                detail=f"{n_missing} rows ({fraction:.1%}) have no target value",
                affected_rows=n_missing,
                fraction=round(fraction, 6),
            )
        )
        df = df.filter(~missing_mask)
        changed = True
        remedies.append(
            TriageRemedy(
                action=TriageAction.DROP_UNLABELED_ROWS,
                column=target_column,
                affected_rows=n_missing,
                justification=(
                    "rows without a target label cannot teach a supervised model; "
                    f"dropped {n_missing} of {rows_before} rows ({fraction:.1%})"
                ),
            )
        )
        if fraction > WARN_DROP_FRACTION:
            _log.warning(
                "triage_heavy_label_loss",
                column=target_column,
                dropped=n_missing,
                fraction=round(fraction, 4),
                hint="over 40% of rows had no label — verify the target column choice",
            )

    # -- 3. Exact-duplicate rows -------------------------------------------------
    # Row-local and safe pre-split BY DEFINITION: dropping a byte-identical
    # duplicate cannot leak anything a single copy of that row didn't already
    # carry. It MUST happen here, before the split — the cleaner's dedup op
    # runs post-split on train only, so if it were the only place duplicates
    # were removed, an identical row could land in both train and holdout and
    # the model would "predict" it from having memorized the training copy,
    # inflating holdout metrics with a form of leakage the split can't catch.
    n_before_dedup = len(df)
    df = df.unique(maintain_order=True, keep="first")
    n_dupes = n_before_dedup - len(df)
    if n_dupes > 0:
        fraction = n_dupes / n_before_dedup
        issues.append(
            TriageIssue(
                code=TriageIssueCode.DUPLICATE_ROWS,
                column="*",
                detail=f"{n_dupes} exact-duplicate rows ({fraction:.1%})",
                affected_rows=n_dupes,
                fraction=round(fraction, 6),
            )
        )
        changed = True
        remedies.append(
            TriageRemedy(
                action=TriageAction.DROP_DUPLICATE_ROWS,
                column="*",
                affected_rows=n_dupes,
                justification=(
                    "exact-duplicate rows removed pre-split so an identical row "
                    "can never appear in both train and holdout; dropped "
                    f"{n_dupes} of {n_before_dedup} rows ({fraction:.1%})"
                ),
            )
        )

    # -- 4. Enough labeled data left? ------------------------------------------
    if len(df) < MIN_LABELED_ROWS:
        issues.append(
            TriageIssue(
                code=TriageIssueCode.TOO_FEW_LABELED_ROWS,
                column=target_column,
                detail=f"only {len(df)} labeled rows remain",
                affected_rows=len(df),
                fraction=1.0,
            )
        )
        raise ValueError(
            f"only {len(df)} labeled rows remain after triage (minimum "
            f"{MIN_LABELED_ROWS}) — '{target_column}' has too few usable labels "
            "to train anything meaningful. Verify the target column or supply "
            "more labeled data."
        )

    # -- 5. Feature columns that are mostly numeric but polluted with junk -----
    for column, dtype in df.schema.items():
        if column == target_column or dtype != pl.String:
            continue
        present = _present_mask(df, column)
        rate, n_junk = _parse_rate(df, column, present)
        if rate >= FEATURE_PARSE_RATE_MIN:
            issues.append(
                TriageIssue(
                    code=TriageIssueCode.NUMERIC_STRING_COLUMN,
                    column=column,
                    detail=f"{rate:.1%} of present values are numeric; {n_junk} junk value(s)",
                    affected_rows=n_junk,
                    fraction=round(1.0 - rate, 6),
                )
            )
            df = df.with_columns(_numeric_parse_expr(column).alias(column))
            changed = True
            remedies.append(
                TriageRemedy(
                    action=TriageAction.COERCE_COLUMN_NUMERIC,
                    column=column,
                    affected_rows=n_junk,
                    justification=(
                        f"column is {rate:.1%} numeric — coerced to numbers; {n_junk} "
                        "junk value(s) became null for the in-pipeline imputer"
                    ),
                )
            )

    # -- 5. Persist -------------------------------------------------------------
    if changed:
        triaged = store_dataframe(
            store,
            df,
            role=DatasetRole.RAW,
            producer_agent="data-triage",
            run_id=run_id,
            parent_ids=(dataset_id,),
            target_column=target_column,
        )
    else:
        triaged = dataset

    report = new_artifact(
        TriageReport,
        producer_agent="data-triage",
        run_id=run_id,
        parent_ids=(dataset_id,) if not changed else (dataset_id, triaged.meta.artifact_id),
        dataset_ref=dataset.ref(),
        triaged_dataset_ref=triaged.ref(),
        issues=tuple(issues),
        remedies=tuple(remedies),
        rows_before=rows_before,
        rows_after=len(df),
    )
    store.put(report)
    _log.info(
        "data_triage_complete",
        issues=len(issues),
        remedies=[f"{r.action.value}:{r.column}" for r in remedies],
        rows_before=rows_before,
        rows_after=len(df),
    )
    return triaged, report
