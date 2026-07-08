"""JUDGMENT_ROADMAP.md Task 5: cross-fitted target encoding.

"Leakage proof, not leakage hope" (the roadmap's own words): the critical
test here is that shuffling the target so the categorical carries NO real
signal must not let the encoded model beat the dummy floor. Cross-fitting
inside fit_transform is what prevents the classic target-encoding leak
(a row's own label bleeding into its own encoded value); this test is what
actually pins that property, rather than assuming the wiring is correct
because it compiles.

Building this trap's leakage test first surfaced a real confound, not a
target-encoding bug: benchmarks/judgment_traps.py's high_cardinality_
categorical generator was only 2 columns (category, binary target) with
~27 rows/category, so most rows were EXACT duplicates of a same-category-
same-label row. Triage's (correct, independently-tested) dedup collapsed
~4000 rows to ~240 unique (category, label) pairs before training ever saw
them -- at which point ordinary small-sample noise (a handful of the now-
≤150 categories landing on a single label by chance) looked like strong
signal to ANY encoder. Fixed by adding a real noise_feature column to the
generator so exact-duplicate rows become astronomically unlikely and dedup
is a no-op -- see that file's docstring for the full account.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from benchmarks.judgment_traps import high_cardinality_categorical

from mlteam.agents.trainer import _column_roles
from mlteam.contracts import TaskType
from mlteam.orchestration import run_tabular_pipeline
from mlteam.store import GuardedArtifactStore, LocalArtifactStore
from mlteam.templates.registry import (
    _TARGET_ENCODE_MIN_CARDINALITY,
    build_preprocessor,
    build_scaled_preprocessor,
)

FAST_LGBM = {"n_estimators": 30}


def _run(tmp_path: Path, df: pl.DataFrame, run_id: str):
    csv_path = tmp_path / "data.csv"
    df.write_csv(csv_path)
    store_root = tmp_path / "store"
    store = GuardedArtifactStore(LocalArtifactStore(store_root), db_path=store_root / "guard.db")
    result = run_tabular_pipeline(
        store,
        csv_path,
        target_column="target",
        task_type=TaskType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
        run_id=run_id,
        seed=42,
        n_folds=3,
        include_tabpfn=False,
        lgbm_config=FAST_LGBM,
    )
    return store, result


def test_column_roles_classifies_high_cardinality_columns():
    df = pl.DataFrame(
        {
            "low_card": [f"a{i % 5}" for i in range(200)],
            "high_card": [f"b{i}" for i in range(200)],  # 200 unique values
            "target": list(range(200)),
        }
    )
    roles = _column_roles(df, "target")
    assert "low_card" in roles.categorical
    assert "high_card" in roles.categorical
    assert "high_card" in roles.high_cardinality_categorical
    assert "low_card" not in roles.high_cardinality_categorical
    # Sanity check on the threshold itself, so this test breaks loudly if
    # the constant ever changes without the assertions above being revisited.
    assert _TARGET_ENCODE_MIN_CARDINALITY == 20


def test_build_preprocessor_routes_high_cardinality_to_target_encoder():
    from sklearn.preprocessing import TargetEncoder

    from mlteam.templates.registry import ColumnRoles

    roles = ColumnRoles(
        numeric=("num1",),
        categorical=("low_card", "high_card"),
        high_cardinality_categorical=("high_card",),
    )
    for builder in (build_preprocessor, build_scaled_preprocessor):
        pre = builder(roles)
        names = {name for name, _, _ in pre.transformers}
        assert "target_encoded" in names
        te_pipeline = next(t for name, t, _ in pre.transformers if name == "target_encoded")
        assert isinstance(te_pipeline.named_steps["encode"], TargetEncoder)
        # The low-cardinality column must NOT be routed to the target encoder.
        _, _, high_card_cols = next(t for t in pre.transformers if t[0] == "target_encoded")
        assert high_card_cols == ["high_card"]


def test_shuffled_target_does_not_beat_dummy_beyond_noise(tmp_path):
    """Calibration: with the target shuffled (category carries zero real
    signal), the encoded LightGBM's roc_auc measured 0.487 -- indistinguishable
    from the 0.5 dummy floor. A generous +/-0.05 band absorbs ordinary fold
    noise while still catching a real leak (which showed roc_auc ~0.9 before
    the generator fix documented in this file's module docstring)."""
    df = high_cardinality_categorical()
    shuffled = df.with_columns(df["target"].shuffle(seed=99))
    _, result = _run(tmp_path, shuffled, run_id="trap-hc-shuffled")

    dummy_auc = result.train_results["exp-dummy"].metrics["roc_auc"].mean
    lgbm_auc = result.train_results["exp-lightgbm"].metrics["roc_auc"].mean
    assert abs(lgbm_auc - dummy_auc) < 0.05


def test_real_signal_still_beats_dummy_with_target_encoding(tmp_path):
    """Positive control for the test above: the SAME generator, unshuffled,
    must clearly beat dummy -- proves the leakage-proof test isn't passing
    merely because target encoding never helps at all."""
    df = high_cardinality_categorical()
    _, result = _run(tmp_path, df, run_id="trap-hc-real")

    dummy_auc = result.train_results["exp-dummy"].metrics["roc_auc"].mean
    lgbm_auc = result.train_results["exp-lightgbm"].metrics["roc_auc"].mean
    assert lgbm_auc - dummy_auc > 0.2


def test_no_duplicate_row_collapse_from_the_generator_fix(tmp_path):
    """Regression guard for the confound this task's leakage test surfaced:
    triage's dedup must be a near no-op on this generator now (holdout_
    fraction=0.2 of 4000 rows -> ~3200 train rows), not collapse it to a
    few hundred unique (category, label) pairs."""
    df = high_cardinality_categorical()
    store, result = _run(tmp_path, df, run_id="trap-hc-dedup-guard")
    from mlteam.agents.data_io import read_dataset

    cleaned = read_dataset(store, store.get(result.cleaned_dataset_id))
    assert cleaned.height > 3000
