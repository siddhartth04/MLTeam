"""Shared result containers passed between training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from mlteam.contracts import MetricSummary, TrainedModel


@dataclass(frozen=True)
class TrainResult:
    """A trained model artifact plus its CV metric summaries."""

    model: TrainedModel
    metrics: dict[str, MetricSummary]
