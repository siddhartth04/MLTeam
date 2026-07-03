"""Serving-layer tests: inference schema, app template, Dockerfile."""

from __future__ import annotations

import ast

import pandas as pd
import pytest

from mlteam.agents import ingest_file, profile_dataset
from mlteam.serving import (
    build_inference_schema_spec,
    load_spec,
    render_app,
    render_dockerfile,
    render_requirements,
    serialize_spec,
    spec_to_pandera,
)

RUN = "test-serving"


@pytest.fixture()
def profile(guarded_store, churn_csv):
    dataset = ingest_file(guarded_store, churn_csv, target_column="churned", run_id=RUN)
    return profile_dataset(guarded_store, dataset.meta.artifact_id, run_id=RUN)


def test_schema_spec_excludes_target(profile):
    spec = build_inference_schema_spec(
        profile,
        target_column="churned",
        feature_names=["tenure_months", "monthly_charges", "contract_type", "churned"],
    )
    names = {c["name"] for c in spec["columns"]}
    assert "churned" not in names
    assert "tenure_months" in names


def test_schema_marks_missing_columns_nullable(profile):
    spec = build_inference_schema_spec(
        profile,
        target_column="churned",
        feature_names=["monthly_charges", "tenure_months"],
    )
    by_name = {c["name"]: c for c in spec["columns"]}
    # monthly_charges has injected nulls in the fixture; tenure has none.
    assert by_name["monthly_charges"]["nullable"] is True
    assert by_name["tenure_months"]["nullable"] is False


def test_schema_roundtrips_and_validates(profile):
    spec = build_inference_schema_spec(
        profile, target_column="churned", feature_names=["tenure_months", "contract_type"]
    )
    reloaded = load_spec(serialize_spec(spec))
    schema = spec_to_pandera(reloaded)
    good = pd.DataFrame({"tenure_months": [12.0], "contract_type": ["monthly"]})
    validated = schema.validate(good)
    assert list(validated.columns) == ["tenure_months", "contract_type"]


def test_generated_app_is_valid_python():
    from mlteam.contracts import TaskType

    source = render_app(TaskType.BINARY_CLASSIFICATION)
    ast.parse(source)  # must be syntactically valid
    assert "schema.validate" in source  # same-schema enforcement present
    assert "predict_proba" in source  # calibrated probabilities returned
    assert 'TASK_TYPE = "binary_classification"' in source


def test_dockerfile_and_requirements():
    dockerfile = render_dockerfile()
    assert "uvicorn" in dockerfile and "serve:app" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    reqs = render_requirements()
    assert "fastapi" in reqs and "pandera" in reqs
