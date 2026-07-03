"""Data cleaning: rule-based proposer + deterministic declarative executor.

Phase 1 proposes ops from hardcoded rules over the DataProfile. In Phase 3 an
LLM takes over proposing (from the same whitelist); the executor is shared
and never changes: ordered declarative ops, applied to produce a NEW dataset,
with justifications and affected-row counts recorded in a CleaningManifest.

Learned transforms (imputation, encoding, scaling) are deliberately NOT here:
they leak if fit outside CV folds, so they live inside the model pipelines.
Only structural, non-learned ops are applied at the dataset level.
"""

from __future__ import annotations

import polars as pl
import structlog

from mlteam.agents.data_io import read_dataset, store_dataframe
from mlteam.contracts import (
    CleaningManifest,
    CleaningOp,
    CleaningOpType,
    DataProfile,
    DatasetRole,
    DatasetVersion,
    WarningCode,
    new_artifact,
)
from mlteam.store.interface import ArtifactStore

_log = structlog.get_logger(__name__)

_MISSING_TOKENS = ("", "na", "n/a", "null", "none", "nan", "?")


def propose_cleaning_ops(profile: DataProfile) -> tuple[CleaningOp, ...]:
    """Hardcoded sensible ops derived from profile warnings (Phase 1 rules)."""
    ops: list[CleaningOp] = []
    string_columns = [c.name for c in profile.columns if c.dtype == "String"]
    if string_columns:
        ops.append(
            CleaningOp(
                op_type=CleaningOpType.STANDARDIZE_MISSING_TOKENS,
                params={"columns": string_columns, "tokens": list(_MISSING_TOKENS)},
                justification="normalize textual missing-value tokens to real nulls",
                affected_columns=tuple(string_columns),
            )
        )
    for warning in profile.warnings:
        if warning.code is WarningCode.CONSTANT_COLUMN and warning.column:
            ops.append(
                CleaningOp(
                    op_type=CleaningOpType.DROP_COLUMN,
                    params={"column": warning.column},
                    justification="constant column carries no predictive signal",
                    affected_columns=(warning.column,),
                )
            )
        elif warning.code is WarningCode.SUSPECTED_ID and warning.column:
            ops.append(
                CleaningOp(
                    op_type=CleaningOpType.DROP_COLUMN,
                    params={"column": warning.column},
                    justification="unique-per-row identifier; keeping it invites memorization",
                    affected_columns=(warning.column,),
                )
            )
        elif warning.code is WarningCode.DUPLICATE_ROWS:
            ops.append(
                CleaningOp(
                    op_type=CleaningOpType.DROP_DUPLICATE_ROWS,
                    params={},
                    justification="exact duplicates bias CV by appearing in multiple folds",
                )
            )
    return tuple(ops)


def _apply_op(df: pl.DataFrame, op: CleaningOp) -> tuple[pl.DataFrame, int]:
    """Apply one whitelisted op; returns (new frame, affected row count)."""
    if op.op_type is CleaningOpType.DROP_COLUMN:
        column = str(op.params["column"])
        return (df.drop(column) if column in df.columns else df), 0
    if op.op_type is CleaningOpType.DROP_DUPLICATE_ROWS:
        deduped = df.unique(maintain_order=True)
        return deduped, df.height - deduped.height
    if op.op_type is CleaningOpType.STANDARDIZE_MISSING_TOKENS:
        columns = [c for c in op.params["columns"] if c in df.columns]
        tokens = [str(t) for t in op.params["tokens"]]
        before = sum(df[c].null_count() for c in columns)
        df = df.with_columns(
            [
                pl.when(pl.col(c).str.strip_chars().str.to_lowercase().is_in(tokens))
                .then(None)
                .otherwise(pl.col(c))
                .alias(c)
                for c in columns
            ]
        )
        after = sum(df[c].null_count() for c in columns)
        return df, after - before
    raise ValueError(f"cleaning op '{op.op_type}' has no deterministic executor yet")


def execute_cleaning(
    store: ArtifactStore,
    dataset_id: str,
    ops: tuple[CleaningOp, ...],
    *,
    run_id: str,
) -> tuple[DatasetVersion, CleaningManifest]:
    """Apply ops in order to produce a NEW cleaned dataset + manifest.

    The input dataset is never mutated; the manifest is replayable on any
    schema-compatible dataset (e.g. the holdout at final-reporting time).
    """
    dataset = store.get(dataset_id)
    df = read_dataset(store, dataset)

    executed: list[CleaningOp] = []
    for op in ops:
        df, affected_rows = _apply_op(df, op)
        executed.append(op.model_copy(update={"affected_rows": affected_rows}))
        _log.info(
            "cleaning_op_applied",
            op=op.op_type.value,
            affected_rows=affected_rows,
            justification=op.justification,
        )

    cleaned = store_dataframe(
        store,
        df,
        role=DatasetRole.CLEANED,
        producer_agent="data-cleaner",
        run_id=run_id,
        parent_ids=(dataset_id,),
        target_column=dataset.target_column,
    )
    manifest = new_artifact(
        CleaningManifest,
        producer_agent="data-cleaner",
        run_id=run_id,
        parent_ids=(dataset_id, cleaned.meta.artifact_id),
        input_dataset_ref=dataset.ref(),
        output_dataset_ref=cleaned.ref(),
        ops=tuple(executed),
    )
    store.put(manifest)
    return cleaned, manifest
