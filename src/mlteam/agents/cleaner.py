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

from typing import Any

import polars as pl
import structlog
from pydantic import BaseModel
from pydantic import Field as PydanticField

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
from mlteam.llm.client import LLMClient
from mlteam.llm.summaries import profile_summary
from mlteam.store.interface import ArtifactStore

_log = structlog.get_logger(__name__)

_MISSING_TOKENS = ("", "na", "n/a", "null", "none", "nan", "?")

# Ops the deterministic dataset-level executor can apply today. Learned
# transforms (imputation, clipping) are deliberately excluded — they belong
# inside CV-fitted pipelines, and proposing them here is a leakage bug.
EXECUTABLE_OP_TYPES: frozenset[CleaningOpType] = frozenset(
    {
        CleaningOpType.DROP_COLUMN,
        CleaningOpType.DROP_DUPLICATE_ROWS,
        CleaningOpType.STANDARDIZE_MISSING_TOKENS,
    }
)


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


CLEANING_PROPOSER_PROMPT = """You are the Data Cleaning agent of an autonomous ML \
engineering team. Given a dataset profile, propose structural cleaning operations.

Rules:
- You may ONLY use these op types (anything else is rejected): drop_column, \
drop_duplicate_rows, standardize_missing_tokens.
- Imputation, clipping, encoding and any transform that LEARNS from data are \
forbidden here — they are fit inside cross-validation pipelines later.
- Drop columns that are constant, identifiers, or flagged as suspected target \
leaks. Never drop the target column.
- Every op needs a specific, factual justification grounded in the profile.
- Propose the minimal set of ops; an empty list is valid if the data is clean.
Return only JSON matching the schema."""


class CleaningOpDraft(BaseModel):
    """Typed op parameters — the structured-output grammar cannot express
    open dicts, so each executable op's parameters are explicit fields."""

    op_type: CleaningOpType
    column: str | None = None  # drop_column
    columns: list[str] = PydanticField(default_factory=list)  # standardize_missing_tokens
    tokens: list[str] = PydanticField(default_factory=list)  # standardize_missing_tokens
    justification: str
    affected_columns: list[str] = PydanticField(default_factory=list)

    def to_params(self) -> dict[str, Any]:
        if self.op_type is CleaningOpType.DROP_COLUMN:
            return {"column": self.column}
        if self.op_type is CleaningOpType.STANDARDIZE_MISSING_TOKENS:
            return {
                "columns": list(self.columns),
                "tokens": list(self.tokens) or list(_MISSING_TOKENS),
            }
        return {}


class CleaningOpsDraft(BaseModel):
    ops: list[CleaningOpDraft] = PydanticField(default_factory=list)


def propose_cleaning_ops_llm(
    client: LLMClient,
    profile: DataProfile,
    *,
    target_column: str | None,
) -> tuple[CleaningOp, ...]:
    """LLM proposes ops from the whitelist; deterministic checks gate them."""

    def _post_validate(draft: CleaningOpsDraft) -> None:
        for op in draft.ops:
            if op.op_type not in EXECUTABLE_OP_TYPES:
                allowed = sorted(t.value for t in EXECUTABLE_OP_TYPES)
                raise ValueError(
                    f"op_type '{op.op_type.value}' is not executable at the dataset "
                    f"level; allowed: {allowed}"
                )
            if not op.justification.strip():
                raise ValueError("every cleaning op requires a non-empty justification")
            if op.op_type is CleaningOpType.DROP_COLUMN:
                if not op.column:
                    raise ValueError("drop_column requires the 'column' field")
                if target_column is not None and op.column == target_column:
                    raise ValueError(f"refusing to drop the target column '{target_column}'")
            if op.op_type is CleaningOpType.STANDARDIZE_MISSING_TOKENS and not op.columns:
                raise ValueError("standardize_missing_tokens requires 'columns'")

    result = client.structured(
        agent="cleaning-proposer",
        system_prompt=CLEANING_PROPOSER_PROMPT,
        user_payload={"profile": profile_summary(profile), "target_column": target_column},
        schema=CleaningOpsDraft,
        post_validate=_post_validate,
    )
    return tuple(
        CleaningOp(
            op_type=op.op_type,
            params=op.to_params(),
            justification=op.justification,
            affected_columns=tuple(op.affected_columns),
        )
        for op in result.output.ops
    )


def _apply_op(df: pl.DataFrame, op: CleaningOp) -> tuple[pl.DataFrame, int]:
    """Apply one whitelisted op; returns (new frame, affected row count)."""
    if op.op_type is CleaningOpType.DROP_COLUMN:
        column = str(op.params["column"])
        if column not in df.columns:
            _log.warning(
                "drop_column_missing",
                column=column,
                hint="manifest op no-oped; schema may have drifted since proposal",
            )
            return df, 0
        return df.drop(column), 0
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


def apply_cleaning_ops_to_frame(df: pl.DataFrame, ops: tuple[CleaningOp, ...]) -> pl.DataFrame:
    """Replay a stored CleaningManifest's ops on an arbitrary frame.

    Used to re-derive the training-time preprocessing on the frozen holdout
    at report time — without this, holdout rows keep raw values (e.g. a
    literal "N/A" string) that training data had normalized to real nulls,
    so the model's encoder treats them as an unseen category: training-
    serving skew inside the platform's OWN evaluation.

    Deliberately skips ``drop_duplicate_rows``: that op's purpose is to keep
    duplicate rows from appearing in multiple CV folds during TRAINING, not
    to change what the evaluation POPULATION looks like. The holdout is a
    fixed, already-frozen sample: whatever natural duplicates it contains
    are part of the true out-of-sample distribution the report should
    describe, not something to prune post hoc. ``standardize_missing_tokens``
    and ``drop_column`` are both purely row-local/schema transforms with no
    such population-level side effect, so they replay safely and are
    leakage-free by the same argument as ``apply_feature_plan_to_frame``.
    """
    for op in ops:
        if op.op_type is CleaningOpType.DROP_DUPLICATE_ROWS:
            continue
        df, _affected_rows = _apply_op(df, op)
    return df


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
