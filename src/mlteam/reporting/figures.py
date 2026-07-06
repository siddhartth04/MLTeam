"""Plotly figure helpers: build JSON-serializable figures and store them.

Plots are the ONLY plotting library used (Plotly), stored as JSON blobs so the
dashboard and HTML reports can both render them without re-running any compute.
"""

from __future__ import annotations

from typing import Any

from mlteam.store.interface import ArtifactStore


def store_figure(store: ArtifactStore, figure: Any) -> str:
    """Serialize a Plotly figure to JSON and store it as a blob; returns hash."""
    import plotly.io as pio

    payload = pio.to_json(figure).encode("utf-8")
    return store.put_blob(payload)


def load_figure_json(store: ArtifactStore, blob_hash: str) -> str:
    """Return the stored Plotly figure JSON string."""
    return store.blob_path(blob_hash).read_text(encoding="utf-8")


def bar_figure(labels: list[str], values: list[float], *, title: str, x_title: str) -> Any:
    """Horizontal bar chart (used for importances), highest at the top."""
    import plotly.graph_objects as go

    order = sorted(range(len(values)), key=lambda i: values[i])
    fig = go.Figure(
        go.Bar(
            x=[values[i] for i in order],
            y=[labels[i] for i in order],
            orientation="h",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="feature",
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def line_figure(x: list[float], y: list[float], *, title: str, x_title: str) -> Any:
    """Line chart (used for partial dependence)."""
    import plotly.graph_objects as go

    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers"))
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="partial dependence",
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def histogram_figure(values: list[float], *, title: str, x_title: str) -> Any:
    """Histogram (used for the residual distribution)."""
    import plotly.graph_objects as go

    fig = go.Figure(go.Histogram(x=values, nbinsx=40))
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="count",
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def scatter_figure(
    x: list[float], y: list[float], *, title: str, x_title: str, y_title: str
) -> Any:
    """Scatter chart (used for residuals vs predicted)."""
    import plotly.graph_objects as go

    fig = go.Figure(go.Scatter(x=x, y=y, mode="markers", marker=dict(size=4, opacity=0.5)))
    fig.add_hline(y=0.0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig
