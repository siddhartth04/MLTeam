"""Typed artifact contracts — the only currency agents exchange.

``ARTIFACT_REGISTRY`` maps artifact type names to classes so the store can
deserialize any artifact without knowing concrete types up front.
"""

from mlteam.contracts.base import (
    SCHEMA_VERSION,
    Artifact,
    ArtifactMeta,
    ArtifactRef,
    canonical_json,
    content_hash,
    file_content_hash,
    new_artifact,
    utcnow,
)
from mlteam.contracts.cleaning import CleaningManifest, CleaningOp, CleaningOpType
from mlteam.contracts.critique import CritiqueReport, Finding, FindingSeverity, Verdict
from mlteam.contracts.dataset import DatasetRole, DatasetVersion
from mlteam.contracts.decision import DecisionRecord
from mlteam.contracts.deployment import DeploymentBundle
from mlteam.contracts.evaluation import (
    BaselineDelta,
    CalibrationSummary,
    EvaluationReport,
    LabelIssueSummary,
    LeakageCheck,
    MetricSummary,
    SliceMetric,
)
from mlteam.contracts.experiments import (
    BudgetEstimate,
    ExcludedTemplate,
    ExperimentPlan,
    ExperimentSpec,
)
from mlteam.contracts.explainability import (
    ExplainabilityReport,
    FeatureImportance,
    FigureRef,
    ImportanceMethod,
    ResidualSlice,
)
from mlteam.contracts.features import FeatureOpType, FeaturePlan, FeatureProposal, FeatureSet
from mlteam.contracts.insight import InsightNarrative
from mlteam.contracts.model import TrainedModel
from mlteam.contracts.outcome import ExperimentOutcome, ExperimentState
from mlteam.contracts.problem_spec import CostAsymmetry, ProblemSpec, TaskType, is_classification
from mlteam.contracts.profile import (
    ColumnProfile,
    CorrelationPair,
    DataProfile,
    ProfileWarning,
    TargetRelationship,
    WarningCode,
    WarningSeverity,
)
from mlteam.contracts.report import FinalReport
from mlteam.contracts.split import SplitPlan, SplitStrategy
from mlteam.contracts.trace import RunTrace, TraceEvent, TraceEventType
from mlteam.contracts.triage import (
    TriageAction,
    TriageIssue,
    TriageIssueCode,
    TriageRemedy,
    TriageReport,
)

ARTIFACT_REGISTRY: dict[str, type[Artifact]] = {
    cls.artifact_type(): cls
    for cls in (
        ProblemSpec,
        DatasetVersion,
        DataProfile,
        SplitPlan,
        CleaningManifest,
        FeaturePlan,
        FeatureSet,
        ExperimentPlan,
        TrainedModel,
        EvaluationReport,
        ExplainabilityReport,
        CritiqueReport,
        DeploymentBundle,
        FinalReport,
        RunTrace,
        TriageReport,
        ExperimentOutcome,
        InsightNarrative,
    )
}

__all__ = [
    "ARTIFACT_REGISTRY",
    "SCHEMA_VERSION",
    "Artifact",
    "ArtifactMeta",
    "ArtifactRef",
    "BaselineDelta",
    "BudgetEstimate",
    "CalibrationSummary",
    "CleaningManifest",
    "CleaningOp",
    "CleaningOpType",
    "ColumnProfile",
    "CorrelationPair",
    "CostAsymmetry",
    "CritiqueReport",
    "DataProfile",
    "DatasetRole",
    "DatasetVersion",
    "DecisionRecord",
    "DeploymentBundle",
    "EvaluationReport",
    "ExcludedTemplate",
    "ExperimentOutcome",
    "ExperimentPlan",
    "ExperimentSpec",
    "ExperimentState",
    "ExplainabilityReport",
    "FeatureImportance",
    "FeatureOpType",
    "FeaturePlan",
    "FeatureProposal",
    "FeatureSet",
    "FigureRef",
    "FinalReport",
    "Finding",
    "FindingSeverity",
    "ImportanceMethod",
    "InsightNarrative",
    "LabelIssueSummary",
    "LeakageCheck",
    "MetricSummary",
    "ProblemSpec",
    "ProfileWarning",
    "ResidualSlice",
    "RunTrace",
    "SliceMetric",
    "SplitPlan",
    "SplitStrategy",
    "TargetRelationship",
    "TaskType",
    "is_classification",
    "TraceEvent",
    "TraceEventType",
    "TrainedModel",
    "TriageAction",
    "TriageIssue",
    "TriageIssueCode",
    "TriageRemedy",
    "TriageReport",
    "Verdict",
    "WarningCode",
    "WarningSeverity",
    "canonical_json",
    "content_hash",
    "file_content_hash",
    "new_artifact",
    "utcnow",
]
