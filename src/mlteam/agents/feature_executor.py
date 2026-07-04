"""Feature-plan executor: deterministic, leakage-honest op application.

Executes the LLM's FeatureProposals against the cleaned TRAIN dataset and
records the outcome as a FeatureSet artifact. The whitelist here is stricter
than the proposal whitelist ON PURPOSE:

- Row-local stateless ops execute (ratio, difference, product, log_transform,
  datetime_decompose, text_length, binning WITH explicit edges): each output
  value depends only on its own row, so computing them once pre-CV cannot
  move information between folds.
- Ops that learn statistics from the data are DROPPED LOUDLY, never silently
  leaked: target_encode (learns from y), frequency_encode / group_aggregate
  (learn X distributions across fold boundaries), binning without explicit
  edges (learns cutpoints). one_hot/ordinal encodes are dropped as redundant —
  every template's in-pipeline preprocessor already encodes categoricals
  inside CV folds.

Every drop is logged with its reason and recorded in
``FeatureSet.dropped_proposals`` so the dashboard shows exactly what the
agent proposed vs. what actually ran.
"""

from __future__ import annotations

import polars as pl
import structlog

from mlteam.agents.data_io import read_dataset, store_dataframe
from mlteam.contracts import (
    DatasetRole,
    FeatureOpType,
    FeaturePlan,
    FeatureProposal,
    FeatureSet,
    new_artifact,
)
from mlteam.contracts.dataset import DatasetVersion
from mlteam.store.interface import ArtifactStore

_log = structlog.get_logger(__name__)

_STATS_OPS = {
    FeatureOpType.TARGET_ENCODE: "learns target statistics — leaks y across CV folds",
    FeatureOpType.FREQUENCY_ENCODE: "learns category frequencies across fold boundaries",
    FeatureOpType.GROUP_AGGREGATE: "learns group statistics across fold boundaries",
}
_REDUNDANT_OPS = {
    FeatureOpType.ONE_HOT_ENCODE: "in-pipeline preprocessor already one-hot/ordinal encodes",
    FeatureOpType.ORDINAL_ENCODE: "in-pipeline preprocessor already ordinal encodes",
}


class _OpRejected(ValueError):
    """Raised by op builders when a proposal cannot be executed safely."""


def _require_columns(df: pl.DataFrame, proposal: FeatureProposal, n: int) -> None:
    if len(proposal.input_columns) != n:
        raise _OpRejected(f"{proposal.op_type.value} needs exactly {n} input column(s)")
    missing = [c for c in proposal.input_columns if c not in df.columns]
    if missing:
        raise _OpRejected(f"input columns not in dataset: {missing}")


def _require_numeric(df: pl.DataFrame, cols: tuple[str, ...]) -> None:
    bad = [c for c in cols if not df[c].dtype.is_numeric()]
    if bad:
        raise _OpRejected(f"non-numeric input columns: {bad}")


def _apply_op(df: pl.DataFrame, proposal: FeatureProposal) -> list[pl.Expr]:
    """Return the expressions that add this proposal's column(s)."""
    op = proposal.op_type
    name = proposal.name

    if op in _STATS_OPS:
        raise _OpRejected(_STATS_OPS[op])
    if op in _REDUNDANT_OPS:
        raise _OpRejected(_REDUNDANT_OPS[op])

    if op is FeatureOpType.RATIO:
        _require_columns(df, proposal, 2)
        _require_numeric(df, proposal.input_columns)
        a, b = (pl.col(c) for c in proposal.input_columns)
        return [pl.when(b != 0).then(a / b).otherwise(None).alias(name)]

    if op is FeatureOpType.DIFFERENCE:
        _require_columns(df, proposal, 2)
        _require_numeric(df, proposal.input_columns)
        a, b = (pl.col(c) for c in proposal.input_columns)
        return [(a - b).alias(name)]

    if op is FeatureOpType.PRODUCT:
        _require_columns(df, proposal, 2)
        _require_numeric(df, proposal.input_columns)
        a, b = (pl.col(c) for c in proposal.input_columns)
        return [(a * b).alias(name)]

    if op is FeatureOpType.LOG_TRANSFORM:
        _require_columns(df, proposal, 1)
        _require_numeric(df, proposal.input_columns)
        c = pl.col(proposal.input_columns[0])
        # log1p is defined for > -1; out-of-domain values become null and the
        # in-pipeline imputer handles them.
        return [pl.when(c > -1).then((c.cast(pl.Float64)).log1p()).otherwise(None).alias(name)]

    if op is FeatureOpType.BINNING:
        _require_columns(df, proposal, 1)
        _require_numeric(df, proposal.input_columns)
        edges = proposal.params.get("edges")
        if not isinstance(edges, list) or len(edges) < 1:
            raise _OpRejected(
                "binning without explicit 'edges' learns cutpoints from data "
                "(fold-unsafe); provide params={'edges': [...]}"
            )
        # edges come from the LLM's free-form params — never trust the types.
        if not all(isinstance(e, int | float) and not isinstance(e, bool) for e in edges):
            raise _OpRejected(f"binning edges must all be numbers, got {edges!r}")
        c = pl.col(proposal.input_columns[0])
        return [c.cut(breaks=[float(e) for e in edges]).cast(pl.String).alias(name)]

    if op is FeatureOpType.DATETIME_DECOMPOSE:
        _require_columns(df, proposal, 1)
        col = proposal.input_columns[0]
        dtype = df.schema[col]
        if dtype not in (pl.Date, pl.Datetime):
            raise _OpRejected(f"'{col}' is {dtype}, not Date/Datetime")
        t = pl.col(col)
        exprs = [
            t.dt.year().alias(f"{name}_year"),
            t.dt.month().alias(f"{name}_month"),
            t.dt.day().alias(f"{name}_day"),
            t.dt.weekday().alias(f"{name}_dayofweek"),
        ]
        if dtype == pl.Datetime:
            exprs.append(t.dt.hour().alias(f"{name}_hour"))
        return exprs

    if op is FeatureOpType.TEXT_LENGTH:
        _require_columns(df, proposal, 1)
        col = proposal.input_columns[0]
        if df.schema[col] != pl.String:
            raise _OpRejected(f"'{col}' is {df.schema[col]}, not a text column")
        return [pl.col(col).str.len_chars().alias(name)]

    raise _OpRejected(f"unsupported op type '{op.value}'")  # future enum values


def execute_feature_plan(
    store: ArtifactStore,
    dataset_id: str,
    plan: FeaturePlan,
    *,
    run_id: str,
) -> tuple[DatasetVersion | None, FeatureSet]:
    """Apply the plan's executable proposals; store the featured dataset.

    Returns ``(featured_dataset, feature_set)``. ``featured_dataset`` is None
    when nothing was applicable — the caller keeps training on the input
    dataset, and the FeatureSet still records why every proposal was dropped.
    """
    dataset = store.get(dataset_id)
    df = read_dataset(store, dataset)
    existing = set(df.columns)

    applied: list[str] = []
    dropped: list[str] = []
    exprs: list[pl.Expr] = []
    new_names: set[str] = set()
    for proposal in plan.proposals:
        try:
            candidate = _apply_op(df, proposal)
            out_names = [e.meta.output_name() for e in candidate]
            clash = [n for n in out_names if n in existing or n in new_names]
            if clash:
                raise _OpRejected(f"output column(s) already exist: {clash}")
        except _OpRejected as exc:
            _log.warning(
                "feature_proposal_dropped",
                feature=proposal.name,
                op=proposal.op_type.value,
                reason=str(exc),
            )
            dropped.append(proposal.name)
            continue
        exprs.extend(candidate)
        new_names.update(out_names)
        applied.append(proposal.name)

    featured_ds: DatasetVersion | None = None
    dataset_ref = dataset.ref()
    if exprs:
        featured = df.with_columns(exprs)
        featured_ds = store_dataframe(
            store,
            featured,
            role=DatasetRole.FEATURES,
            producer_agent="feature-executor",
            run_id=run_id,
            parent_ids=(dataset_id, plan.meta.artifact_id),
            target_column=dataset.target_column,
        )
        dataset_ref = featured_ds.ref()
        _log.info(
            "feature_plan_executed",
            applied=applied,
            dropped=dropped,
            new_columns=sorted(new_names),
        )
    else:
        _log.warning("feature_plan_nothing_applicable", dropped=dropped)

    feature_set = new_artifact(
        FeatureSet,
        producer_agent="feature-executor",
        run_id=run_id,
        parent_ids=(plan.meta.artifact_id, dataset_id),
        plan_ref=plan.ref(),
        dataset_ref=dataset_ref,
        feature_names=tuple(sorted(new_names)),
        dropped_proposals=tuple(dropped),
    )
    store.put(feature_set)
    return featured_ds, feature_set
