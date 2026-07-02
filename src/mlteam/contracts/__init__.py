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
from mlteam.contracts.experiments import BudgetEstimate, ExperimentPlan, ExperimentSpec
from mlteam.contracts.features import FeatureOpType, FeaturePlan, FeatureProposal, FeatureSet
from mlteam.contracts.model import TrainedModel
from mlteam.contracts.problem_spec import CostAsymmetry, ProblemSpec, TaskType
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
        CritiqueReport,
        DeploymentBundle,
        FinalReport,
        RunTrace,
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
    "DeploymentBundle",
    "EvaluationReport",
    "ExperimentPlan",
    "ExperimentSpec",
    "FeatureOpType",
    "FeaturePlan",
    "FeatureProposal",
    "FeatureSet",
    "FinalReport",
    "Finding",
    "FindingSeverity",
    "LabelIssueSummary",
    "LeakageCheck",
    "MetricSummary",
    "ProblemSpec",
    "ProfileWarning",
    "RunTrace",
    "SliceMetric",
    "SplitPlan",
    "SplitStrategy",
    "TargetRelationship",
    "TaskType",
    "TraceEvent",
    "TraceEventType",
    "TrainedModel",
    "Verdict",
    "WarningCode",
    "WarningSeverity",
    "canonical_json",
    "content_hash",
    "file_content_hash",
    "new_artifact",
    "utcnow",
]
