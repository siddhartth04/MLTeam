"""Judgment trap suite: senior-engineer calls as executable specs.

"Behaves like a senior ML engineer" is unfalsifiable without a benchmark.
Each trap here is a small, deterministic dataset engineered to require ONE
specific senior-engineer judgment call -- the kind a first-year Kaggle
competitor makes reflexively and MLTeam, as evaluated 2026-07-08, could not
(see JUDGMENT_ROADMAP.md's background section for the live incident this
suite formalizes). Traps whose capability doesn't exist yet are wired to
`xfail(strict=True)` in tests/test_judgment_traps.py, keyed to the roadmap
task that closes them -- when a task lands, its trap's xfail marker must be
removed in that same commit, or the strict xfail turns into a hard error.

Generators are pure functions (seeded numpy/polars, no I/O) so trap data
never depends on network access or a cached benchmarks/data/ file the way
the OpenML-backed accuracy benchmarks in datasets.py do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from mlteam.contracts import TaskType

_SEED = 20260108  # the date of the founding incident this suite formalizes


@dataclass(frozen=True)
class TrapSpec:
    """One judgment trap: a dataset plus the senior-engineer call it demands."""

    name: str
    target_column: str
    task_type: TaskType
    expected: str  # one-line description of the senior call
    # "Task N" if a roadmap task closes it, "deferred" if no task is
    # assigned yet (still xfail-enforced, not silently must-pass), or
    # None if it must pass today.
    blocked_by: str | None


def skewed_price_target(n: int = 3000, seed: int = _SEED) -> pl.DataFrame:
    """A price-like target spanning >=3 orders of magnitude, lognormally
    distributed -- the exact shape of the 2026-07-08 house-prices target
    (₹100K-₹14B) that no code path could transform. Two informative
    features (area, quality) drive the log-scale target linearly.

    Calibration note (against real LightGBM fits, not a toy model): RMSE is
    the wrong metric to calibrate on here -- it's dominated by the same
    tail outliers whether or not the target is log-transformed, so a raw
    fit and a log1p fit score almost identically on it. MAE is where the
    log1p advantage is real and grows with skew: at noise std 0.4 the
    improvement was ~3%; at 0.8 (used below) plain-sklearn LightGBM MAE
    dropped from ~381k to ~347k raw->log1p (~10% better), so this is the
    metric and noise level the trap's test calibrates its threshold on.
    """
    rng = np.random.default_rng(seed)
    area = rng.uniform(300, 4000, n)
    quality = rng.integers(1, 11, n)
    log_price = 10.0 + 0.0008 * area + 0.15 * quality + rng.normal(0, 0.8, n)
    price = np.expm1(log_price)  # spans ~3 orders of magnitude with the noise tail
    return pl.DataFrame(
        {
            "area_sqft": area,
            "quality_score": quality,
            "price": price,
        }
    )


def leaky_feature(n: int = 1000, seed: int = _SEED) -> pl.DataFrame:
    """One feature is the target plus tiny noise -- the near-perfect-MI
    leakage battery must catch it, or the cleaner must drop it outright."""
    rng = np.random.default_rng(seed)
    target = rng.integers(0, 2, n)
    leak = target + rng.normal(0, 0.001, n)
    noise_feature = rng.normal(0, 1, n)
    return pl.DataFrame(
        {
            "leaky_echo": leak,
            "genuine_noise": noise_feature,
            "target": target,
        }
    )


def id_memorization(n: int = 500, seed: int = _SEED) -> pl.DataFrame:
    """A unique-per-row string identifier -- must be dropped as a suspected
    identifier, never trained on (it invites memorization, not learning)."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "customer_id": [f"CUST-{i:06d}" for i in range(n)],
            "tenure_months": rng.integers(1, 72, n).astype(float),
            "monthly_spend": np.round(rng.uniform(10, 200, n), 2),
            "churned": rng.integers(0, 2, n),
        }
    )


def duplicate_rows(
    n_unique: int = 300, dup_fraction: float = 0.2, seed: int = _SEED
) -> pl.DataFrame:
    """20% exact-duplicate rows -- triage must dedupe before any split, or
    an identical row can appear in both train and holdout."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n_unique)
    y = (x > 0).astype(int)
    base = pl.DataFrame({"feature": x, "target": y})
    n_dupes = int(n_unique * dup_fraction / (1 - dup_fraction))
    return pl.concat([base, base.head(n_dupes)])


def placeholder_labels(
    n: int = 1000, placeholder_fraction: float = 0.1, seed: int = _SEED
) -> pl.DataFrame:
    """A numeric target column where 10% of values are the marketplace
    placeholder string "Call for Price" -- regression-protects the
    2026-07-07 triage fix (TARGET_PARSE_RATE_MIN 0.95 -> 0.80, junk values
    named in errors/remedies) against the exact incident that motivated it."""
    rng = np.random.default_rng(seed)
    n_placeholder = int(n * placeholder_fraction)
    prices = [f"{rng.uniform(1, 50):.1f} Lac" for _ in range(n - n_placeholder)]
    prices += ["Call for Price"] * n_placeholder
    rng.shuffle(prices)
    return pl.DataFrame(
        {
            "area_sqft": rng.uniform(300, 3000, n),
            "price": prices,
        }
    )


def high_cardinality_categorical(
    n: int = 4000, n_levels: int = 150, seed: int = _SEED
) -> pl.DataFrame:
    """A 150-level categorical carrying most of the signal via its
    per-level mean -- one-hot/ordinal encoding cannot usefully exploit 150
    sparse levels; only a fold-safe target/frequency encoding can. See
    tests/test_judgment_traps.py for how the pass threshold was calibrated
    (with-column ceiling vs without-column floor).

    ``noise_feature`` is a real column, not just noise added to the logit --
    without it, this frame is only 2 columns (category, binary target) with
    ~27 rows/category, so most rows are EXACT duplicates of a same-category-
    same-label row. Triage's (correct, independently-tested) dedup then
    collapses ~4000 rows to ~240 unique (category, label) pairs before
    training ever sees them, at which point ordinary small-sample noise --
    a handful of the now-≤150 categories landing on a single label by pure
    chance -- looks like strong signal to any encoder, confounding the
    leakage-proof test this trap exists to support. A continuous per-row
    noise column makes exact duplicates astronomically unlikely, so dedup
    is a no-op here and the trap tests target encoding, not the interaction
    between two unrelated triage steps.
    """
    rng = np.random.default_rng(seed)
    level_effects = rng.normal(0, 2.0, n_levels)
    levels = rng.integers(0, n_levels, n)
    noise = rng.normal(0, 0.5, n)
    logit = level_effects[levels] + noise
    target = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pl.DataFrame(
        {
            "high_card_category": [f"cat_{lv:04d}" for lv in levels],
            "noise_feature": rng.normal(0, 1, n),
            "target": target,
        }
    )


def temporal_drift_disguised(n: int = 1000, seed: int = _SEED) -> pl.DataFrame:
    """Rows are time-ordered with genuine target drift (a regime shift
    partway through), a date column present, but the task is presented as
    plain REGRESSION -- random K-fold CV silently overestimates real-world
    performance here because it lets the model see "future" rows in
    training folds that predict "past" test rows. No task currently closes
    this trap (see JUDGMENT_ROADMAP.md's deferred section: adversarial
    validation / temporal-risk detection needs real design)."""
    from datetime import datetime, timedelta

    rng = np.random.default_rng(seed)
    start = datetime(2023, 1, 1)
    dates = [start + timedelta(days=int(d)) for d in range(n)]
    x = rng.normal(0, 1, n)
    # Regime shift at the midpoint: the x->y relationship inverts, so a
    # model trained on a random mix of both regimes learns an averaged,
    # wrong relationship that random CV cannot detect but a true
    # chronological holdout would expose immediately.
    regime = (np.arange(n) >= n // 2).astype(float)
    y = np.where(regime == 0, 2.0 * x, -2.0 * x) + rng.normal(0, 0.3, n)
    return pl.DataFrame({"event_date": dates, "x": x, "y": y})


def constant_target(n: int = 200, seed: int = _SEED) -> pl.DataFrame:
    """Zero-variance target -- there is nothing to model. The pipeline must
    fail LOUDLY, naming the target column, not die deep inside sklearn with
    an opaque error (e.g. a division-by-zero in a metric computed over a
    single-valued array)."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "feature": rng.normal(0, 1, n),
            "constant_price": np.full(n, 42.0),
        }
    )


TRAPS: tuple[TrapSpec, ...] = (
    TrapSpec(
        name="skewed_price_target",
        target_column="price",
        task_type=TaskType.REGRESSION,
        expected="log1p-transform a price-like target spanning >=3 orders of magnitude",
        blocked_by=None,  # closed by JUDGMENT_ROADMAP.md Task 2
    ),
    TrapSpec(
        name="leaky_feature",
        target_column="target",
        task_type=TaskType.BINARY_CLASSIFICATION,
        expected="detect or drop a near-perfect-MI leaked feature",
        blocked_by=None,
    ),
    TrapSpec(
        name="id_memorization",
        target_column="churned",
        task_type=TaskType.BINARY_CLASSIFICATION,
        expected="drop a unique-per-row identifier column before training",
        blocked_by=None,
    ),
    TrapSpec(
        name="duplicate_rows",
        target_column="target",
        task_type=TaskType.BINARY_CLASSIFICATION,
        expected="dedupe exact-duplicate rows before splitting",
        blocked_by=None,
    ),
    TrapSpec(
        name="placeholder_labels",
        target_column="price",
        task_type=TaskType.REGRESSION,
        expected="coerce a mostly-numeric target and name the placeholder junk values",
        blocked_by=None,
    ),
    TrapSpec(
        name="high_cardinality_categorical",
        target_column="target",
        task_type=TaskType.BINARY_CLASSIFICATION,
        expected="cross-fitted target/frequency encoding for a 150-level categorical",
        blocked_by=None,  # closed by JUDGMENT_ROADMAP.md Task 5
    ),
    TrapSpec(
        name="temporal_drift_disguised",
        target_column="y",
        task_type=TaskType.REGRESSION,
        expected="warn that random CV overestimates under a disguised regime shift",
        # Not "None" (must-pass-today): no task closes this yet -- see
        # JUDGMENT_ROADMAP.md's deferred section (adversarial validation /
        # temporal-risk detection needs real design). "deferred" pins the
        # gap as a documented, xfail-enforced fact instead of silently
        # treating it as passing.
        blocked_by="deferred",
    ),
    TrapSpec(
        name="constant_target",
        target_column="constant_price",
        task_type=TaskType.REGRESSION,
        expected="fail loudly, naming the target column, on a zero-variance target",
        blocked_by=None,
    ),
)

_GENERATORS = {
    "skewed_price_target": skewed_price_target,
    "leaky_feature": leaky_feature,
    "id_memorization": id_memorization,
    "duplicate_rows": duplicate_rows,
    "placeholder_labels": placeholder_labels,
    "high_cardinality_categorical": high_cardinality_categorical,
    "temporal_drift_disguised": temporal_drift_disguised,
    "constant_target": constant_target,
}


def generate(trap: TrapSpec) -> pl.DataFrame:
    return _GENERATORS[trap.name]()


def get_trap(name: str) -> TrapSpec:
    for trap in TRAPS:
        if trap.name == name:
            return trap
    raise KeyError(f"unknown trap '{name}'; known: {[t.name for t in TRAPS]}")
