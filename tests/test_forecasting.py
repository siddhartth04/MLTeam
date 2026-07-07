"""Forecasting path: temporal splits, lag features, and the E2E pipeline."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from mlteam.agents import ingest_file, plan_split
from mlteam.agents.splitter import detect_time_column
from mlteam.agents.time_features import add_forecast_features, resolve_horizon
from mlteam.agents.trainer import _make_cv
from mlteam.contracts import SplitStrategy, TaskType

RUN = "test-forecast-run"
N_DAYS = 160


def make_series_frame(n: int = N_DAYS, seed: int = 11) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(n)]
    trend = np.linspace(50, 90, n)
    weekly = 8.0 * np.sin(2 * np.pi * np.arange(n) / 7.0)
    sales = trend + weekly + rng.normal(scale=2.0, size=n)
    return pl.DataFrame(
        {
            "date": days,
            "promo": rng.integers(0, 2, size=n),
            "sales": np.round(sales, 3),
        }
    )


@pytest.fixture()
def series_csv(tmp_path: Path) -> Path:
    path = tmp_path / "sales.csv"
    make_series_frame().write_csv(path)
    return path


# -- time features -----------------------------------------------------------


def test_detect_time_column_single():
    assert detect_time_column(make_series_frame()) == "date"


def test_detect_time_column_none_raises():
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match="has none"):
        detect_time_column(df)


def test_detect_time_column_ambiguous_raises():
    df = make_series_frame().with_columns(pl.col("date").alias("shipped_at"))
    with pytest.raises(ValueError, match="ambiguous"):
        detect_time_column(df)


def test_add_forecast_features_columns_and_warmup():
    df = make_series_frame()
    out = add_forecast_features(df, time_column="date", target_column="sales")
    for col in (
        "sales_lag_1",
        "sales_lag_7",
        "sales_rollmean_3",
        "sales_rollstd_7",
        "date_year",
        "date_month",
        "date_dayofweek",
    ):
        assert col in out.columns, f"missing {col}"
    # warmup rows with null lags are dropped, none survive
    assert len(out) < len(df)
    assert out.null_count().sum_horizontal().item() == 0
    # output is time-sorted
    dates = out["date"].to_list()
    assert dates == sorted(dates)


def test_add_forecast_features_lag_values_are_past_only():
    df = make_series_frame(n=30)
    out = add_forecast_features(df, time_column="date", target_column="sales")
    sales_by_date = dict(zip(df["date"].to_list(), df["sales"].to_list(), strict=True))
    for row in out.head(5).iter_rows(named=True):
        prev_day = row["date"] - timedelta(days=1)
        assert row["sales_lag_1"] == pytest.approx(sales_by_date[prev_day])


def test_add_forecast_features_horizon_1_is_byte_identical_to_default():
    df = make_series_frame()
    explicit = add_forecast_features(
        df, time_column="date", target_column="sales", horizon=1
    )
    default = add_forecast_features(df, time_column="date", target_column="sales")
    assert explicit.equals(default)


def test_add_forecast_features_horizon_7_drops_unrealizable_close_lags():
    df = make_series_frame()
    out = add_forecast_features(
        df, time_column="date", target_column="sales", horizon=7
    )
    for col in ("sales_lag_1", "sales_lag_2", "sales_lag_3"):
        assert col not in out.columns, f"{col} should not exist at horizon=7"
    assert "sales_lag_7" in out.columns
    # rolling windows must also start no closer than the horizon
    assert out.null_count().sum_horizontal().item() == 0


def test_add_forecast_features_horizon_matches_lag_shift():
    """At horizon=h, sales_lag_h on row t must equal the true value h days back."""
    df = make_series_frame(n=40)
    horizon = 3
    out = add_forecast_features(
        df, time_column="date", target_column="sales", horizon=horizon
    )
    sales_by_date = dict(zip(df["date"].to_list(), df["sales"].to_list(), strict=True))
    for row in out.head(5).iter_rows(named=True):
        origin_minus_h = row["date"] - timedelta(days=horizon)
        assert row["sales_lag_3"] == pytest.approx(sales_by_date[origin_minus_h])


def test_add_forecast_features_horizon_beyond_vetted_lags_fails_loudly():
    with pytest.raises(ValueError, match="exceeds every vetted lag"):
        add_forecast_features(
            make_series_frame(), time_column="date", target_column="sales", horizon=30
        )


def test_add_forecast_features_rejects_invalid_horizon():
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        add_forecast_features(
            make_series_frame(), time_column="date", target_column="sales", horizon=0
        )


def test_resolve_horizon_parses_leading_integer():
    assert resolve_horizon("7 days") == 7
    assert resolve_horizon("next 14 days") == 14
    assert resolve_horizon("1") == 1


def test_resolve_horizon_defaults_to_one_step_when_unparseable():
    assert resolve_horizon(None) == 1
    assert resolve_horizon("next month") == 1
    assert resolve_horizon("") == 1


def test_add_forecast_features_rejects_non_numeric_target():
    df = make_series_frame().with_columns(pl.col("sales").cast(pl.String))
    with pytest.raises(ValueError, match="must be numeric"):
        add_forecast_features(df, time_column="date", target_column="sales")


def test_add_forecast_features_rejects_too_short_series():
    with pytest.raises(ValueError, match="too short"):
        add_forecast_features(
            make_series_frame(n=12), time_column="date", target_column="sales"
        )


def test_add_forecast_features_rejects_panel_data():
    """Duplicate timestamps would let entity B's value leak into A's lag."""
    single = make_series_frame(n=40)
    panel = pl.concat([single, single.with_columns(pl.col("sales") * 2)]).sort("date")
    with pytest.raises(ValueError, match="duplicate timestamps"):
        add_forecast_features(panel, time_column="date", target_column="sales")


# -- temporal split ----------------------------------------------------------


def test_forecasting_uses_temporal_strategy(local_store, series_csv):
    raw = ingest_file(local_store, series_csv, target_column="sales", run_id=RUN)
    plan = plan_split(
        local_store,
        raw.meta.artifact_id,
        task_type=TaskType.FORECASTING,
        run_id=RUN,
        seed=42,
    )
    assert plan.strategy is SplitStrategy.TEMPORAL
    assert plan.time_column == "date"


def test_temporal_holdout_is_chronological_tail(local_store, series_csv):
    from mlteam.agents.data_io import read_dataset

    raw = ingest_file(local_store, series_csv, target_column="sales", run_id=RUN)
    plan = plan_split(
        local_store,
        raw.meta.artifact_id,
        task_type=TaskType.FORECASTING,
        run_id=RUN,
        seed=42,
        holdout_fraction=0.2,
    )
    train = read_dataset(local_store, local_store.get(plan.train_ref.artifact_id))
    holdout = read_dataset(local_store, local_store.get(plan.holdout_ref.artifact_id))
    # every training row is strictly earlier than every holdout row
    assert train["date"].max() < holdout["date"].min()
    # train stays time-sorted so TimeSeriesSplit CV is valid on row order
    assert train["date"].to_list() == sorted(train["date"].to_list())
    assert len(holdout) == round(0.2 * N_DAYS)


def test_temporal_split_never_bisects_a_timestamp(local_store, tmp_path):
    """Rows sharing the boundary timestamp must all land in holdout."""
    from mlteam.agents.data_io import read_dataset

    single = make_series_frame(n=60)
    panel = pl.concat([single, single.with_columns(pl.col("sales") * 2)]).sort("date")
    path = tmp_path / "panel.csv"
    panel.write_csv(path)
    raw = ingest_file(local_store, path, target_column="sales", run_id=RUN)
    plan = plan_split(
        local_store,
        raw.meta.artifact_id,
        task_type=TaskType.FORECASTING,
        run_id=RUN,
        seed=42,
    )
    train = read_dataset(local_store, local_store.get(plan.train_ref.artifact_id))
    holdout = read_dataset(local_store, local_store.get(plan.holdout_ref.artifact_id))
    assert train["date"].max() < holdout["date"].min()


def test_temporal_split_rejects_too_few_folds_rows(local_store, tmp_path):
    path = tmp_path / "tiny.csv"
    make_series_frame(n=16).write_csv(path)
    raw = ingest_file(local_store, path, target_column="sales", run_id=RUN)
    with pytest.raises(ValueError, match="reduce n_folds"):
        plan_split(
            local_store,
            raw.meta.artifact_id,
            task_type=TaskType.FORECASTING,
            run_id=RUN,
            seed=42,
            holdout_fraction=0.4,
            n_folds=10,
        )


def test_forecasting_cv_is_time_series_split():
    from sklearn.model_selection import TimeSeriesSplit

    cv = _make_cv(TaskType.FORECASTING, n_folds=4, seed=42)
    assert isinstance(cv, TimeSeriesSplit)


# -- end to end ---------------------------------------------------------------


def test_forecasting_pipeline_end_to_end(guarded_store, series_csv):
    from mlteam.orchestration.pipeline import run_tabular_pipeline

    result = run_tabular_pipeline(
        guarded_store,
        series_csv,
        target_column="sales",
        task_type=TaskType.FORECASTING,
        primary_metric="rmse",
        seed=42,
        n_folds=3,
        include_tabpfn=False,
    )
    split = guarded_store.get(result.split_plan_id)
    assert split.strategy is SplitStrategy.TEMPORAL
    assert split.time_column == "date"
    # lag features flowed into training data
    from mlteam.agents.data_io import read_dataset

    cleaned = read_dataset(guarded_store, guarded_store.get(result.cleaned_dataset_id))
    assert any(c.startswith("sales_lag_") for c in cleaned.columns)
    # with real lag features LightGBM must crush the mean-predicting dummy
    report = result.evaluation_report
    rmse_delta = next(
        d for d in report.baseline_deltas if d.metric == "rmse" and "dummy" in d.baseline_name
    )
    assert rmse_delta.model_value < rmse_delta.baseline_value


def test_forecasting_pipeline_honors_declared_horizon(guarded_store, series_csv):
    """A 7-step horizon must reach the feature engineer through the pipeline
    entrypoint, not just the unit-level add_forecast_features call."""
    from mlteam.orchestration.pipeline import run_tabular_pipeline

    result = run_tabular_pipeline(
        guarded_store,
        series_csv,
        target_column="sales",
        task_type=TaskType.FORECASTING,
        primary_metric="rmse",
        seed=42,
        n_folds=3,
        include_tabpfn=False,
        forecast_horizon=7,
    )
    from mlteam.agents.data_io import read_dataset

    cleaned = read_dataset(guarded_store, guarded_store.get(result.cleaned_dataset_id))
    for col in ("sales_lag_1", "sales_lag_2", "sales_lag_3"):
        assert col not in cleaned.columns, f"{col} unrealizable at horizon=7"
    assert "sales_lag_7" in cleaned.columns


def test_final_report_renders_forecast_horizon_line(guarded_store, series_csv):
    from mlteam.agents import critique_evaluation
    from mlteam.contracts import ProblemSpec, new_artifact
    from mlteam.llm import FixtureTransport, LLMClient
    from mlteam.orchestration.pipeline import run_tabular_pipeline
    from mlteam.reporting import ReportInputs, build_final_report

    result = run_tabular_pipeline(
        guarded_store,
        series_csv,
        target_column="sales",
        task_type=TaskType.FORECASTING,
        primary_metric="rmse",
        seed=42,
        n_folds=3,
        include_tabpfn=False,
        forecast_horizon=3,
    )
    model = next(
        r.model for k, r in result.train_results.items() if "dummy" not in k
    )
    problem = new_artifact(
        ProblemSpec,
        producer_agent="test",
        run_id=RUN,
        task_type=TaskType.FORECASTING,
        target_column="sales",
        primary_metric="rmse",
        prediction_horizon="3 days",
        time_column="date",
        human_confirmed=True,
    )
    critique = critique_evaluation(
        LLMClient(
            FixtureTransport.from_texts(
                '{"findings": [], "verdict": "approve", "iteration_actions": []}'
            ),
            model="claude-sonnet-4-6",
        ),
        guarded_store,
        report=result.evaluation_report,
        primary_metric="rmse",
        run_id=RUN,
    )
    inputs = ReportInputs(
        problem=problem,
        model=model,
        evaluation=result.evaluation_report,
        critique=critique,
        split_plan=guarded_store.get(result.split_plan_id),
        goal_text="forecast sales 3 days ahead",
        best_template_id=result.best_template_id,
        best_experiment_id=model.experiment_id,
    )
    final = build_final_report(guarded_store, inputs, run_id=RUN)
    technical = guarded_store.blob_path(final.technical_report_blob).read_text(
        encoding="utf-8"
    )
    assert "3-step-ahead" in technical
