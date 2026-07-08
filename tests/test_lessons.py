"""JUDGMENT_ROADMAP.md Task 7: lessons substrate v0.

The ledger records WHICH template won; nothing recorded WHAT a run taught
("log-transform price-like targets" -- the founding incident's lesson had
nowhere to live). This suite covers the JSONL round-trip, the retro
agent's best-effort recording (validated lessons kept, malformed ones
rejected loudly without failing the run, and nothing at all recorded for
a fixture-transport run -- the same policy as the outcome ledger), and
that the planner's prompt payload includes only same-task-type lessons.
"""

from __future__ import annotations

import json

from mlteam.agents.cleaner import propose_cleaning_ops_llm
from mlteam.agents.experiment_planner import plan_experiments
from mlteam.agents.ingest import ingest_file
from mlteam.agents.profiler import profile_dataset
from mlteam.agents.retro import run_retro
from mlteam.contracts import (
    CritiqueReport,
    ProblemSpec,
    TaskType,
    Verdict,
    new_artifact,
)
from mlteam.llm import FixtureTransport, LLMClient
from mlteam.store.lessons import matching_lessons, read_lessons, record_lesson

RUN = "test-lessons"


def _client(*texts: str, max_attempts: int = 3) -> tuple[LLMClient, FixtureTransport]:
    transport = FixtureTransport.from_texts(*texts)
    client = LLMClient(transport, model="claude-sonnet-4-6", max_attempts=max_attempts)
    return client, transport


def _live(client: LLMClient, monkeypatch) -> None:
    """FixtureTransport is never "live" by design (see LLMClient.is_live) --
    monkeypatched here specifically to test retro's RECORDING logic in
    isolation from the live-transport gate, which is tested separately."""
    monkeypatch.setattr(type(client), "is_live", property(lambda self: True))


def _critique(verdict: Verdict = Verdict.APPROVE) -> CritiqueReport:
    return new_artifact(
        CritiqueReport,
        producer_agent="test",
        run_id=RUN,
        evaluation_ref={"artifact_id": "e", "artifact_type": "EvaluationReport"},
        verdict=verdict,
        findings=(),
        iteration_actions=(),
    )


def test_lessons_round_trip(tmp_path):
    record_lesson(
        tmp_path,
        lesson="log-transform price-like targets spanning multiple orders of magnitude",
        evidence="MAE was 32% of median target with no transform applied",
        task_type="regression",
        source_run_id="run-1",
    )
    record_lesson(
        tmp_path,
        lesson="cross-fit target encoding for categoricals above ~20 levels",
        evidence="ordinal encoding scored below dummy on a 150-level categorical",
        task_type="binary_classification",
        source_run_id="run-2",
    )
    all_entries = read_lessons(tmp_path)
    assert len(all_entries) == 2
    assert all_entries[0]["source_run_id"] == "run-1"

    regression_only = matching_lessons(tmp_path, task_type="regression")
    assert len(regression_only) == 1
    assert regression_only[0]["source_run_id"] == "run-1"


def test_lessons_file_absent_returns_empty(tmp_path):
    assert read_lessons(tmp_path) == []
    assert matching_lessons(tmp_path, task_type="regression") == []


def test_retro_fixture_transport_records_nothing(tmp_path):
    """The default policy: FixtureTransport is never "live", so retro must
    not write anything -- a fixture replay is not a new data point."""
    client, _ = _client(
        json.dumps(
            {
                "lessons": [
                    {"lesson": "this should never be recorded", "evidence": "n/a"},
                ]
            }
        )
    )
    recorded = run_retro(
        client,
        tmp_path,
        run_id=RUN,
        task_type=TaskType.REGRESSION,
        critique=_critique(),
    )
    assert recorded == ()
    assert read_lessons(tmp_path) == []


def test_retro_records_validated_lessons(tmp_path, monkeypatch):
    client, _ = _client(
        json.dumps(
            {
                "lessons": [
                    {
                        "lesson": "log-transform price-like targets spanning orders of magnitude",
                        "evidence": "MAE was 32% of median target with no transform applied",
                    }
                ]
            }
        )
    )
    _live(client, monkeypatch)
    recorded = run_retro(
        client,
        tmp_path,
        run_id=RUN,
        task_type=TaskType.REGRESSION,
        critique=_critique(),
    )
    assert len(recorded) == 1
    assert "log-transform" in recorded[0]["lesson"]
    on_disk = read_lessons(tmp_path)
    assert len(on_disk) == 1
    assert on_disk[0]["source_run_id"] == RUN


def test_retro_rejects_malformed_lesson_loudly_without_failing(tmp_path, monkeypatch):
    """A lesson over the 300-char wire cap fails schema validation -- retro
    must log it and return empty, never raise to the caller or partially
    record."""
    client, _ = _client(
        json.dumps({"lessons": [{"lesson": "x" * 301, "evidence": "too long"}]}),
        max_attempts=1,
    )
    _live(client, monkeypatch)
    recorded = run_retro(
        client,
        tmp_path,
        run_id=RUN,
        task_type=TaskType.REGRESSION,
        critique=_critique(),
    )
    assert recorded == ()
    assert read_lessons(tmp_path) == []


def test_retro_halt_verdict_also_runs(tmp_path, monkeypatch):
    """Halts teach more -- retro must run for a HALT verdict too, not just
    approve."""
    client, _ = _client(
        json.dumps(
            {
                "lessons": [
                    {
                        "lesson": "investigate leakage before trusting near-perfect CV",
                        "evidence": "x",
                    }
                ]
            }
        )
    )
    _live(client, monkeypatch)
    recorded = run_retro(
        client,
        tmp_path,
        run_id=RUN,
        task_type=TaskType.BINARY_CLASSIFICATION,
        critique=_critique(Verdict.HALT),
    )
    assert len(recorded) == 1


def test_planner_payload_includes_only_matching_task_type_lessons(guarded_store, churn_csv):
    record_lesson(
        guarded_store.root,
        lesson="log-transform price-like targets spanning orders of magnitude",
        evidence="MAE was 32% of median target with no transform applied",
        task_type="regression",
        source_run_id="run-regression",
    )
    record_lesson(
        guarded_store.root,
        lesson="cross-fit target encoding for categoricals above ~20 levels",
        evidence="ordinal encoding scored below dummy on a 150-level categorical",
        task_type="binary_classification",
        source_run_id="run-classification",
    )

    dataset = ingest_file(guarded_store, churn_csv, target_column="churned", run_id=RUN)
    profile = profile_dataset(guarded_store, dataset.meta.artifact_id, run_id=RUN)
    spec = new_artifact(
        ProblemSpec,
        producer_agent="test",
        run_id=RUN,
        task_type=TaskType.BINARY_CLASSIFICATION,
        target_column="churned",
        primary_metric="roc_auc",
    )
    guarded_store.put(spec)

    client, transport = _client(
        json.dumps(
            {
                "experiments": [
                    {"experiment_id": "b1", "template_id": "dummy_baseline", "is_baseline": True},
                    {
                        "experiment_id": "b2",
                        "template_id": "lightgbm_default",
                        "is_baseline": True,
                    },
                ],
                "rationale": "baselines only",
            }
        )
    )
    plan_experiments(
        client,
        guarded_store,
        profile=profile,
        problem=spec,
        n_rows=200,
        run_id=RUN,
        store_root=guarded_store.root,
    )
    sent = json.loads(transport.requests[0].user_content)
    lessons_sent = sent["lessons_from_previous_runs"]
    assert len(lessons_sent) == 1
    assert "cross-fit target encoding" in lessons_sent[0]["lesson"]


def test_cleaning_proposer_payload_includes_only_matching_task_type_lessons(
    guarded_store, churn_csv
):
    record_lesson(
        guarded_store.root,
        lesson="log-transform price-like targets spanning orders of magnitude",
        evidence="MAE was 32% of median target with no transform applied",
        task_type="regression",
        source_run_id="run-regression",
    )
    record_lesson(
        guarded_store.root,
        lesson="cross-fit target encoding for categoricals above ~20 levels",
        evidence="ordinal encoding scored below dummy on a 150-level categorical",
        task_type="binary_classification",
        source_run_id="run-classification",
    )

    dataset = ingest_file(guarded_store, churn_csv, target_column="churned", run_id=RUN)
    profile = profile_dataset(guarded_store, dataset.meta.artifact_id, run_id=RUN)

    client, transport = _client(json.dumps({"ops": []}))
    propose_cleaning_ops_llm(
        client,
        profile,
        target_column="churned",
        store_root=guarded_store.root,
        task_type=TaskType.BINARY_CLASSIFICATION,
    )
    sent = json.loads(transport.requests[0].user_content)
    lessons_sent = sent["lessons_from_previous_runs"]
    assert len(lessons_sent) == 1
    assert "cross-fit target encoding" in lessons_sent[0]["lesson"]
