"""Jinja2 HTML templates for the business and technical reports.

Templates are self-contained (inline CSS, Plotly figures embedded as JSON and
rendered client-side via a single CDN script) so a report is one portable file.
"""

from __future__ import annotations

_BASE_CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; color: #1a1a2e;
  max-width: 960px; margin: 2rem auto; padding: 0 1rem; line-height: 1.55; }
h1 { border-bottom: 3px solid #16213e; padding-bottom: .3rem; }
h2 { color: #16213e; margin-top: 2rem; }
.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 1rem; margin: 1rem 0; }
.metric-card { background: #f4f6fb; border-radius: 10px; padding: 1rem; position: relative; }
.metric-card .value { font-size: 1.6rem; font-weight: 700; color: #0f3460; }
.metric-card .label { font-size: .85rem; color: #555; }
.metric-card--holdout { border-left: 4px solid #1e8449; }
.metric-card--cv { border-left: 4px solid #b9770e; }
.badge { display: inline-block; font-size: .68rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: .03em; padding: .1rem .4rem; border-radius: 4px; margin-bottom: .3rem; }
.badge--holdout { background: #d5f5e3; color: #1e8449; }
.badge--cv { background: #fdebd0; color: #b9770e; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { text-align: left; padding: .5rem .7rem; border-bottom: 1px solid #e0e0e0; }
th { background: #16213e; color: #fff; }
.verdict { display: inline-block; padding: .3rem .8rem; border-radius: 6px;
  font-weight: 700; }
.verdict.approve { background: #d5f5e3; color: #1e8449; }
.verdict.iterate_with_actions { background: #fdebd0; color: #b9770e; }
.verdict.halt { background: #fadbd8; color: #c0392b; }
.pass { color: #1e8449; font-weight: 700; }
.fail { color: #c0392b; font-weight: 700; }
.figure { margin: 1.2rem 0; }
.muted { color: #777; font-size: .85rem; }
"""

_BUSINESS_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{{ title }}</title>
<style>{{ css }}</style>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
</head><body>
<h1>{{ title }}</h1>
<p class="muted">Run {{ run_id }} · generated {{ generated_at }}</p>

<h2>Outcome</h2>
<p>We built a model to <strong>{{ goal }}</strong>. The recommendation is:
<span class="verdict {{ verdict }}">{{ verdict_label }}</span>.</p>
<p>{{ headline_sentence }}</p>

{% if insight_findings %}
<h2>Key findings</h2>
<ul>{% for f in insight_findings %}<li>{{ f }}</li>{% endfor %}</ul>
{% if insight_caveats %}
<p class="muted"><strong>Worth knowing:</strong></p>
<ul class="muted">{% for c in insight_caveats %}<li>{{ c }}</li>{% endfor %}</ul>
{% endif %}
{% endif %}

<h2>Headline results</h2>
<div class="metric-grid">
{% for m in headline_metrics %}
{% set kind_label = 'out-of-sample' if m.kind == 'holdout' else 'cross-validation' %}
  <div class="metric-card metric-card--{{ m.kind }}">
  <div class="badge badge--{{ m.kind }}">{{ kind_label }}</div>
  <div class="value">{{ m.value }}</div>
  <div class="label">{{ m.label }}</div></div>
{% endfor %}
</div>
<p class="muted"><strong>Out-of-sample</strong> cards are measured on a held-out
set the model never saw during development, so they reflect real-world
performance. <strong>Cross-validation</strong> cards are in-sample estimates
from training and tend to run slightly optimistic relative to true holdout
performance — treat them as a secondary signal, not the headline claim.</p>

<h2>What drives the prediction</h2>
{% if top_features %}
<p>The strongest signals the model relies on:</p>
<ul>{% for f in top_features %}<li><strong>{{ f.feature }}</strong></li>{% endfor %}</ul>
{% for fig in business_figures %}
<div class="figure" id="fig-{{ loop.index }}"></div>
<script>Plotly.newPlot('fig-{{ loop.index }}', {{ fig }});</script>
{% endfor %}
{% else %}<p class="muted">Explainability was not generated for this run.</p>{% endif %}

<h2>Trust &amp; safety checks</h2>
<table><tr><th>Check</th><th>Result</th></tr>
{% for c in safety_checks %}
<tr><td>{{ c.name }}</td>
<td class="{{ 'pass' if c.passed else 'fail' }}">{{ 'PASS' if c.passed else 'FAIL' }}</td></tr>
{% endfor %}</table>
</body></html>
"""

_TECHNICAL_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{{ title }}</title>
<style>{{ css }}</style>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
</head><body>
<h1>{{ title }}</h1>
<p class="muted">Run {{ run_id }} · generated {{ generated_at }}</p>

<h2>Problem specification</h2>
<table>
<tr><th>Task type</th><td>{{ problem.task_type }}</td></tr>
<tr><th>Target</th><td>{{ problem.target_column }}</td></tr>
<tr><th>Primary metric</th><td>{{ problem.primary_metric }}</td></tr>
<tr><th>Best model</th><td>{{ best_template }} ({{ best_experiment }})</td></tr>
{% if target_transform is not none %}
<tr><th>Target transform</th><td>{{ target_transform }} — all metrics below (CV and
holdout) are reported in ORIGINAL units; the transform is inverted automatically
at prediction time.</td></tr>
{% endif %}
{% if forecast_horizon is not none %}
<tr><th>Forecast horizon</th><td>{{ forecast_horizon }}-step-ahead — every metric below
(CV and holdout) measures the model's ability to predict {{ forecast_horizon }} step(s)
into the future, not the next single observation.</td></tr>
{% endif %}
</table>

<h2>Cross-validation metrics</h2>
<table><tr><th>Metric</th><th>Mean</th><th>Std</th><th>Folds</th></tr>
{% for m in cv_metrics %}
<tr><td>{{ m.name }}</td><td>{{ m.mean }}</td><td>{{ m.std }}</td>
<td class="muted">{{ m.folds }}</td></tr>
{% endfor %}</table>

<h2>Holdout (out-of-sample) metrics</h2>
{% if holdout_metrics %}
<table><tr><th>Metric</th><th>Value</th><th>95% CI</th><th>Holdout n</th></tr>
{% for m in holdout_metrics %}
<tr><td>{{ m.name }}</td><td>{{ m.value }}</td>
<td class="muted">{% if m.ci_low is defined %}[{{ m.ci_low }},
 {{ m.ci_high }}]{% else %}—{% endif %}</td>
<td class="muted">{{ m.n if m.n is defined else '—' }}</td></tr>
{% endfor %}</table>
<p class="muted">Holdout was frozen at ingestion and unlocked only for this report. The 95%
confidence interval is a seeded bootstrap over the holdout rows (1000 resamples) — with a
small holdout, the point estimate's later decimal places are sampling noise, not precision.</p>
{% else %}<p class="muted">Holdout metrics unavailable.</p>{% endif %}

<h2>Baseline deltas</h2>
<table><tr><th>Baseline</th><th>Metric</th><th>Baseline</th><th>Model</th><th>Delta</th><th>Significant</th><th>Test</th></tr>
{% for d in baseline_deltas %}
<tr><td>{{ d.baseline }}</td><td>{{ d.metric }}</td><td>{{ d.baseline_value }}</td>
<td>{{ d.model_value }}</td><td>{{ d.delta }}</td>
<td class="{{ 'pass' if d.significant else 'muted' }}">{{ d.significant }}</td>
<td class="muted">{{ d.significance_method }}{% if d.p_value is not none %}
 (p={{ d.p_value }}){% endif %}</td></tr>
{% endfor %}</table>

<h2>Leakage checks</h2>
<table><tr><th>Check</th><th>Result</th><th>Details</th></tr>
{% for c in leakage_checks %}
<tr><td>{{ c.name }}</td>
<td class="{{ 'pass' if c.passed else 'fail' }}">{{ 'PASS' if c.passed else 'FAIL' }}</td>
<td class="muted">{{ c.details }}</td></tr>
{% endfor %}</table>

<h2>Calibration</h2>
{% if calibration %}
<table><tr><th>Expected calibration error</th><th>Brier score</th><th>Method</th></tr>
<tr><td>{{ calibration.expected_calibration_error }}</td>
<td>{{ calibration.brier_score }}</td>
<td class="muted">{{ calibration.method }}</td></tr></table>
{% else %}<p class="muted">Not measured for this run.</p>{% endif %}

<h2>Decision threshold</h2>
{% if decision_threshold %}
<p class="muted">Cost-optimal cutoff from out-of-fold probabilities, given a stated cost of
{{ decision_threshold.false_positive_cost }} per false positive and
{{ decision_threshold.false_negative_cost }} per false negative.</p>
<table><tr><th>Threshold</th><th>Expected cost</th><th>Cost at 0.5</th>
<th>TP</th><th>FP</th><th>FN</th><th>TN</th></tr>
<tr><td>{{ decision_threshold.threshold }}</td>
<td>{{ decision_threshold.expected_cost }}</td>
<td class="muted">{{ decision_threshold.cost_at_default_threshold }}</td>
<td>{{ decision_threshold.true_positives }}</td>
<td>{{ decision_threshold.false_positives }}</td>
<td>{{ decision_threshold.false_negatives }}</td>
<td>{{ decision_threshold.true_negatives }}</td></tr>
</table>
{% else %}<p class="muted">Not computed (no stated cost asymmetry, or not applicable to this
task type).</p>{% endif %}

<h2>Critic verdict</h2>
<p><span class="verdict {{ verdict }}">{{ verdict_label }}</span></p>
{% if findings %}<table><tr><th>Severity</th><th>Category</th><th>Evidence</th></tr>
{% for f in findings %}<tr><td>{{ f.severity }}</td><td>{{ f.category }}</td>
<td>{{ f.evidence }}</td></tr>{% endfor %}</table>{% endif %}

<h2>Explainability</h2>
{% for fig in technical_figures %}
<div class="figure" id="tfig-{{ loop.index }}"></div>
<script>Plotly.newPlot('tfig-{{ loop.index }}', {{ fig }});</script>
{% endfor %}

<h2>Iteration history</h2>
{% if iterations %}
<table><tr><th>#</th><th>Verdict</th><th>Best experiment</th><th>New experiments</th></tr>
{% for it in iterations %}<tr><td>{{ it.index }}</td><td>{{ it.verdict }}</td>
<td>{{ it.best }}</td><td class="muted">{{ it.new }}</td></tr>{% endfor %}</table>
{% else %}<p class="muted">No iteration was required.</p>{% endif %}

<h2>Reproducibility</h2>
<table>
<tr><th>Seed</th><td>{{ repro.seed }}</td></tr>
<tr><th>Environment hash</th><td class="muted">{{ repro.env_hash }}</td></tr>
<tr><th>Training data artifact</th><td class="muted">{{ repro.train_data }}</td></tr>
</table>
</body></html>
"""


def business_template() -> str:
    return _BUSINESS_TEMPLATE


def technical_template() -> str:
    return _TECHNICAL_TEMPLATE


def base_css() -> str:
    return _BASE_CSS
