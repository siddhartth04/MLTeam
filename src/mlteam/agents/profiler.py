"""Dataset profiling: deterministic Polars/sklearn computation, no LLM.

Produces a DataProfile with per-column stats, correlations, target
relationships (mutual information + correlation), and structured warnings —
including SUSPECTED_LEAK flags that the leakage checks consume downstream.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from mlteam.agents.data_io import read_dataset
from mlteam.contracts import (
    ColumnProfile,
    CorrelationPair,
    DataProfile,
    ProfileWarning,
    TargetRelationship,
    WarningCode,
    WarningSeverity,
    new_artifact,
)
from mlteam.store.interface import ArtifactStore

_MAX_CORRELATION_PAIRS = 50
_TOP_VALUES = 5
_CLASS_LIKE_MAX_UNIQUE = 20
_HIGH_MISSING_PCT = 30.0
_HIGH_CARDINALITY_FRACTION = 0.5
_ID_UNIQUE_FRACTION = 0.95
_LEAK_MI_THRESHOLD = 0.9
_LEAK_CORR_THRESHOLD = 0.99
_IMBALANCE_MINORITY_FRACTION = 0.10


def _finite(value: object) -> float | None:
    """Coerce to a finite float or None (NaN/Inf are unhashable in payloads)."""
    if value is None:
        return None
    number = float(value)  # type: ignore[arg-type]
    return number if math.isfinite(number) else None


def _is_class_like(series: pl.Series) -> bool:
    if series.dtype in (pl.String, pl.Boolean, pl.Categorical):
        return True
    return series.n_unique() <= _CLASS_LIKE_MAX_UNIQUE


def _column_profile(series: pl.Series, n_rows: int) -> ColumnProfile:
    is_numeric = series.dtype.is_numeric()
    top_values: tuple[tuple[str, int], ...] = ()
    if not is_numeric:
        rows = [
            (str(value), int(count))
            for value, count in series.drop_nulls().value_counts().iter_rows()
        ]
        # Explicit (count desc, value asc) ordering: engine-internal tie-break
        # order must never reach a content-hashed payload.
        rows.sort(key=lambda item: (-item[1], item[0]))
        top_values = tuple(rows[:_TOP_VALUES])
    non_null = series.drop_nulls()
    return ColumnProfile(
        name=series.name,
        dtype=str(series.dtype),
        cardinality=series.n_unique(),
        missing_pct=round(100.0 * series.null_count() / n_rows, 4),
        mean=_finite(series.mean()) if is_numeric else None,
        std=_finite(series.std()) if is_numeric else None,
        min_value=str(non_null.min()) if non_null.len() else None,
        max_value=str(non_null.max()) if non_null.len() else None,
        top_values=top_values,
    )


def _correlation_pairs(df: pl.DataFrame) -> tuple[CorrelationPair, ...]:
    numeric = [c for c in df.columns if df[c].dtype.is_numeric()]
    if len(numeric) < 2:
        return ()
    corr = df.select(numeric).to_pandas().corr(numeric_only=True)
    pairs = []
    for i, col_a in enumerate(numeric):
        for col_b in numeric[i + 1 :]:
            value = _finite(corr.loc[col_a, col_b])
            if value is not None:
                pairs.append(
                    CorrelationPair(
                        column_a=col_a, column_b=col_b, pearson=max(-1.0, min(1.0, value))
                    )
                )
    pairs.sort(key=lambda p: abs(p.pearson), reverse=True)
    return tuple(pairs[:_MAX_CORRELATION_PAIRS])


def _encode_features(df: pl.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Numeric matrix for MI: median-fill numerics, ordinal-code the rest."""
    encoded = []
    for col in feature_cols:
        series = df[col]
        if series.dtype.is_numeric():
            if series.dtype in (pl.Float32, pl.Float64):
                series = series.fill_nan(None)
            fill = series.median() if series.null_count() < series.len() else 0.0
            encoded.append(series.fill_null(fill).to_numpy().astype(np.float64))
        else:
            codes = series.cast(pl.String).fill_null("__missing__").rank("dense").to_numpy()
            encoded.append(codes.astype(np.float64))
    return np.column_stack(encoded)


def _target_entropy(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log(probs)).sum())


def _gaussian_equivalent_correlation(mi_nats: float) -> float:
    """Convert differential-entropy-based MI (nats, unbounded for continuous
    targets) to the Gaussian-equivalent correlation r = sqrt(1 - exp(-2*MI))
    (Joe, 1989 -- the MI of a bivariate Gaussian with correlation r is
    exactly -0.5*ln(1-r^2), so this is that relationship inverted). Bounds
    the score to [0, 1) so it is comparable to Pearson/Spearman and to the
    classification branch's entropy-normalized MI: e.g. r=0.9 -> MI=0.83
    nats, which the OLD unnormalized code would have already flagged at the
    0.9 threshold despite r=0.9 being a legitimate, non-leaky predictor."""
    return math.sqrt(max(0.0, 1.0 - math.exp(-2.0 * mi_nats)))


def _target_relationships(
    df: pl.DataFrame, target_column: str, seed: int
) -> tuple[TargetRelationship, ...]:
    from scipy.stats import spearmanr
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    feature_cols = [c for c in df.columns if c != target_column]
    if not feature_cols:
        return ()
    features = _encode_features(df, feature_cols)
    target_series = df[target_column]
    classification = _is_class_like(target_series)
    if classification:
        y = target_series.cast(pl.String).fill_null("__missing__").rank("dense").to_numpy()
        raw_mi = mutual_info_classif(features, y, random_state=seed)
        entropy = max(_target_entropy(y), 1e-12)
        mi_values = np.clip(raw_mi / entropy, 0.0, 1.0)
    else:
        y = target_series.to_numpy().astype(np.float64)
        raw_mi = mutual_info_regression(features, y, random_state=seed)
        # Regression MI is unbounded (differential entropy) -- normalize to
        # the SAME [0, 1) scale the classification branch uses, so a single
        # 0.9/0.99 threshold means "strong association" in both branches
        # instead of "possible leak" in one and "run-of-the-mill signal" in
        # the other for the identical raw number.
        mi_values = np.array([_gaussian_equivalent_correlation(float(m)) for m in raw_mi])

    y_numeric = y.astype(np.float64)
    relationships = []
    for idx, col in enumerate(feature_cols):
        feature = features[:, idx]
        corr = None
        spearman_corr = None
        if np.std(feature) > 0 and np.std(y_numeric) > 0:
            corr = _finite(np.corrcoef(feature, y_numeric)[0, 1])
            # Spearman catches monotone-but-nonlinear leaks (e.g. exp(y) or a
            # rank/probability transform of the target) that Pearson misses
            # entirely -- a perfect monotone relationship has |pearson| < 1
            # but |spearman| == 1.
            spearman_result = spearmanr(feature, y_numeric)
            spearman_corr = _finite(float(spearman_result.statistic))
        relationships.append(
            TargetRelationship(
                column=col,
                mutual_information=_finite(mi_values[idx]),
                correlation=None if corr is None else max(-1.0, min(1.0, corr)),
                spearman=None if spearman_corr is None else max(-1.0, min(1.0, spearman_corr)),
            )
        )
    return tuple(relationships)


def _column_warnings(df: pl.DataFrame, target_column: str | None) -> list[ProfileWarning]:
    warnings: list[ProfileWarning] = []
    for col in df.columns:
        series = df[col]
        if series.n_unique() <= 1:
            warnings.append(
                ProfileWarning(
                    code=WarningCode.CONSTANT_COLUMN,
                    severity=WarningSeverity.WARNING,
                    column=col,
                    message=f"'{col}' is constant and carries no signal",
                )
            )
        elif (
            col != target_column
            and (series.dtype == pl.String or series.dtype.is_integer())
            and series.n_unique() >= _ID_UNIQUE_FRACTION * df.height
        ):
            warnings.append(
                ProfileWarning(
                    code=WarningCode.SUSPECTED_ID,
                    severity=WarningSeverity.WARNING,
                    column=col,
                    message=f"'{col}' is near-unique per row — likely an identifier",
                )
            )
        elif (
            not series.dtype.is_numeric()
            and series.n_unique() > _HIGH_CARDINALITY_FRACTION * df.height
        ):
            warnings.append(
                ProfileWarning(
                    code=WarningCode.HIGH_CARDINALITY,
                    severity=WarningSeverity.INFO,
                    column=col,
                    message=f"'{col}' has very high cardinality ({series.n_unique()} values)",
                )
            )
        if 100.0 * series.null_count() / df.height > _HIGH_MISSING_PCT:
            warnings.append(
                ProfileWarning(
                    code=WarningCode.HIGH_MISSINGNESS,
                    severity=WarningSeverity.WARNING,
                    column=col,
                    message=f"'{col}' is more than {_HIGH_MISSING_PCT:.0f}% missing",
                )
            )
        if series.dtype in (pl.Date, pl.Datetime):
            warnings.append(
                ProfileWarning(
                    code=WarningCode.DATETIME_DETECTED,
                    severity=WarningSeverity.INFO,
                    column=col,
                    message=f"'{col}' is a datetime column; consider temporal splitting",
                )
            )
    return warnings


def _dataset_warnings(
    df: pl.DataFrame,
    target_column: str | None,
    relationships: tuple[TargetRelationship, ...],
) -> list[ProfileWarning]:
    warnings: list[ProfileWarning] = []
    n_duplicates = df.height - df.unique().height
    if n_duplicates > 0:
        warnings.append(
            ProfileWarning(
                code=WarningCode.DUPLICATE_ROWS,
                severity=WarningSeverity.WARNING,
                column=None,
                message=f"{n_duplicates} exact duplicate rows detected",
            )
        )
    if target_column is not None and _is_class_like(df[target_column]):
        counts = df[target_column].drop_nulls().value_counts()
        # Denominator is LABELED rows (counts sums to n_labeled after
        # drop_nulls), not df.height: dividing by the full height
        # (incl. unlabeled rows awaiting prediction) understates the true
        # minority fraction and can silently mask real imbalance.
        n_labeled = counts["count"].sum()
        minority = counts["count"].min() / n_labeled if n_labeled else 0.0
        if minority < _IMBALANCE_MINORITY_FRACTION:
            warnings.append(
                ProfileWarning(
                    code=WarningCode.TARGET_IMBALANCE,
                    severity=WarningSeverity.WARNING,
                    column=target_column,
                    message=f"minority class is only {100 * minority:.1f}% of rows",
                )
            )
    for rel in relationships:
        mi_leak = rel.mutual_information is not None and rel.mutual_information > _LEAK_MI_THRESHOLD
        corr_leak = rel.correlation is not None and abs(rel.correlation) > _LEAK_CORR_THRESHOLD
        # Spearman catches monotone-but-nonlinear leaks (e.g. a rank or
        # probability transform of the target) that Pearson's linear
        # correlation misses entirely.
        spearman_leak = (
            rel.spearman is not None and abs(rel.spearman) > _LEAK_CORR_THRESHOLD
        )
        if mi_leak or corr_leak or spearman_leak:
            warnings.append(
                ProfileWarning(
                    code=WarningCode.SUSPECTED_LEAK,
                    severity=WarningSeverity.CRITICAL,
                    column=rel.column,
                    message=(
                        f"'{rel.column}' predicts the target almost perfectly "
                        f"(MI={rel.mutual_information}, corr={rel.correlation}, "
                        f"spearman={rel.spearman}) — possible target leakage"
                    ),
                )
            )
    return warnings


def profile_dataset(
    store: ArtifactStore,
    dataset_id: str,
    *,
    run_id: str,
    seed: int = 42,
) -> DataProfile:
    """Compute and store a full DataProfile for a dataset."""
    dataset = store.get(dataset_id)
    df = read_dataset(store, dataset)
    target = dataset.target_column

    columns = tuple(_column_profile(df[c], df.height) for c in df.columns)
    correlations = _correlation_pairs(df)
    relationships = _target_relationships(df, target, seed) if target is not None else ()
    warnings = tuple(_column_warnings(df, target) + _dataset_warnings(df, target, relationships))

    profile = new_artifact(
        DataProfile,
        producer_agent="profiler",
        run_id=run_id,
        parent_ids=(dataset_id,),
        dataset_ref=dataset.ref(),
        columns=columns,
        correlations=correlations,
        target_relationships=relationships,
        warnings=warnings,
    )
    store.put(profile)
    return profile
