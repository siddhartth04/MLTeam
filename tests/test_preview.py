"""Column preview: cheap per-column stats available before Gate 1.

Before this existed, the requirement agent saw column NAMES ONLY and could
not rank target candidates or flag a likely duplicate/leaky column — it was
choosing blind. These tests prove the preview survives real-world junk
(nulls, unicode currency, bucket labels, unparseable dates) without ever
raising, and classifies the common column shapes correctly.
"""

from __future__ import annotations

import json
from pathlib import Path

from mlteam.agents.preview import column_preview
from mlteam.agents.requirement import understand_requirements
from mlteam.llm import FixtureTransport, LLMClient

RUN = "test-preview"


def _write(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows), encoding="utf-8")


def test_preview_classifies_numeric_text_and_date_columns(tmp_path: Path):
    rows = ["balcony,region,signup_date,price"]
    rows += [
        f"{i % 5},north,2024-0{1 + i % 9}-15,{100000 + i * 1000}" for i in range(60)
    ]
    path = tmp_path / "listings.csv"
    _write(path, rows)

    previews = {p.name: p for p in column_preview(path)}
    assert previews["balcony"].inferred_type == "numeric"
    assert previews["region"].inferred_type == "text"
    assert previews["signup_date"].inferred_type == "date"
    assert previews["price"].inferred_type == "numeric"
    assert previews["region"].n_unique == 1
    assert previews["region"].sample_values == ("north",)


def test_preview_never_crashes_on_junk(tmp_path: Path):
    """Unicode currency, bucket labels, blank cells, unparseable dates — the
    exact shapes that used to crash strict CSV/date parsing."""
    rows = ["price,balcony,weird_date,notes"]
    rows += [f"₹{100000 + i * 1000},{i % 5},2024-0{1 + i % 9}-15,ok" for i in range(59)]
    rows.append(",> 10,not-a-date,")  # blank price, bucket label, junk date, blank note
    path = tmp_path / "messy.csv"
    _write(path, rows)

    previews = {p.name: p for p in column_preview(path)}
    assert previews["price"].inferred_type == "numeric"
    assert previews["price"].null_fraction > 0  # the one blank counted
    assert previews["balcony"].inferred_type == "numeric"
    assert previews["weird_date"].inferred_type == "date"  # 59/60 still parse
    assert previews["notes"].inferred_type == "text"


def test_preview_reports_null_fraction_and_cardinality(tmp_path: Path):
    rows = ["id,label"]
    rows += [f"{i},{'yes' if i % 2 == 0 else 'no'}" for i in range(50)]
    rows += [f"{i}," for i in range(50, 60)]  # 10 blank labels
    path = tmp_path / "data.csv"
    _write(path, rows)

    previews = {p.name: p for p in column_preview(path)}
    assert previews["id"].n_unique == 60
    assert previews["label"].null_fraction == round(10 / 60, 4)
    assert previews["label"].n_unique == 2


def test_preview_handles_all_empty_column(tmp_path: Path):
    rows = ["a,b"] + [f"{i}," for i in range(60)]
    path = tmp_path / "empty_col.csv"
    _write(path, rows)

    previews = {p.name: p for p in column_preview(path)}
    assert previews["b"].inferred_type == "empty"
    assert previews["b"].null_fraction == 1.0
    assert previews["b"].n_unique == 0
    assert previews["b"].sample_values == ()


def test_preview_caps_sample_values_at_three(tmp_path: Path):
    rows = ["category"] + [f"cat-{i}" for i in range(60)]
    path = tmp_path / "high_card.csv"
    _write(path, rows)

    previews = column_preview(path)
    assert len(previews[0].sample_values) == 3


def _client(*texts: str) -> tuple[LLMClient, FixtureTransport]:
    transport = FixtureTransport.from_texts(*texts)
    return LLMClient(transport, model="claude-sonnet-4-6"), transport


def test_requirement_agent_receives_column_profile_in_payload(guarded_store, tmp_path: Path):
    rows = ["tenure_months,churned"] + [f"{i},{i % 2}" for i in range(60)]
    path = tmp_path / "churn.csv"
    _write(path, rows)
    previews = column_preview(path)

    client, transport = _client(
        json.dumps(
            {
                "task_type": "binary_classification",
                "target_column": "churned",
                "primary_metric": "roc_auc",
            }
        )
    )
    understand_requirements(
        client,
        guarded_store,
        goal_text="predict churn",
        dataset_columns=["tenure_months", "churned"],
        run_id=RUN,
        column_profile=previews,
    )
    sent = json.loads(transport.requests[0].user_content)
    assert "column_profile" in sent
    sent_names = {c["name"] for c in sent["column_profile"]}
    assert sent_names == {"tenure_months", "churned"}


def test_requirement_agent_omits_column_profile_when_not_given(guarded_store):
    client, transport = _client(
        json.dumps(
            {
                "task_type": "binary_classification",
                "target_column": "churned",
                "primary_metric": "roc_auc",
            }
        )
    )
    understand_requirements(
        client,
        guarded_store,
        goal_text="predict churn",
        dataset_columns=["tenure_months", "churned"],
        run_id=RUN,
    )
    sent = json.loads(transport.requests[0].user_content)
    assert "column_profile" not in sent
