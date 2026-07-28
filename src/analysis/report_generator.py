"""Auto-generates standalone HTML evaluation reports with embedded charts."""
from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Template

from src.utils.logger import get_logger

log = get_logger(__name__)

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ title }}</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 20px; background: #f5f5f5; color: #333; }
  .container { max-width: 1200px; margin: 0 auto; background: white;
               padding: 32px; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,.1); }
  h1 { color: #1a1a2e; border-bottom: 3px solid #4CAF50; padding-bottom: 12px; }
  h2 { color: #16213e; margin-top: 36px; }
  table { border-collapse: collapse; width: 100%; margin: 16px 0; }
  th { background: #4CAF50; color: white; padding: 10px 14px; text-align: left; }
  td { padding: 9px 14px; border-bottom: 1px solid #e0e0e0; }
  tr:hover td { background: #f9fbe7; }
  .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 20px 0; }
  .metric-card { background: #e8f5e9; border-radius: 10px; padding: 16px 20px;
                  text-align: center; }
  .metric-card .value { font-size: 2em; font-weight: bold; color: #2e7d32; }
  .metric-card .label { font-size: 0.85em; color: #555; margin-top: 4px; }
  .chart-img { width: 100%; border-radius: 8px; margin: 12px 0; }
  .badge { display: inline-block; padding: 3px 10px; border-radius: 12px;
            font-size: 0.8em; background: #e3f2fd; color: #1565c0; margin: 2px; }
  .warn { background: #fff3e0; color: #e65100; }
  .section { margin: 32px 0; }
  footer { text-align: center; color: #888; margin-top: 40px; font-size: 0.85em; }
</style>
</head>
<body>
<div class="container">
  <h1>🍦 {{ title }}</h1>
  <p>Generated on <strong>{{ generated_at }}</strong></p>

  <div class="section">
    <h2>Model Summary</h2>
    <table>
      <tr><th>Field</th><th>Value</th></tr>
      {% for k, v in model_summary.items() %}
      <tr><td>{{ k }}</td><td>{{ v }}</td></tr>
      {% endfor %}
    </table>
  </div>

  <div class="section">
    <h2>Key Metrics</h2>
    <div class="metric-grid">
      <div class="metric-card">
        <div class="value">{{ "%.4f"|format(overall.mAP50) }}</div>
        <div class="label">mAP@0.5</div>
      </div>
      <div class="metric-card">
        <div class="value">{{ "%.4f"|format(overall.mAP50_95) }}</div>
        <div class="label">mAP@0.5:0.95</div>
      </div>
      <div class="metric-card">
        <div class="value">{{ "%.2f%%"|format(overall.counting_accuracy * 100) }}</div>
        <div class="label">Counting Accuracy</div>
      </div>
      <div class="metric-card">
        <div class="value">{{ "%.2f"|format(overall.mean_absolute_count_error) }}</div>
        <div class="label">Mean Count Error</div>
      </div>
    </div>
  </div>

  <div class="section">
    <h2>Per-Class Performance</h2>
    <table>
      <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
      {% for cls, m in per_class.items() %}
      <tr>
        <td>{{ cls }}</td>
        <td>{{ "%.4f"|format(m.precision) if m.precision is not none else "N/A" }}</td>
        <td>{{ "%.4f"|format(m.recall) if m.recall is not none else "N/A" }}</td>
        <td>{{ "%.4f"|format(m.f1) if m.f1 is not none else "N/A" }}</td>
      </tr>
      {% endfor %}
    </table>
  </div>

  {% if worst_classes %}
  <div class="section">
    <h2>Worst Performing Classes</h2>
    <p>These classes showed the lowest F1 scores and may benefit from more training data:</p>
    {% for cls in worst_classes %}
    <span class="badge warn">{{ cls }}</span>
    {% endfor %}
  </div>
  {% endif %}

  {% if confusion_matrix_b64 %}
  <div class="section">
    <h2>Confusion Matrix</h2>
    <img class="chart-img" src="data:image/png;base64,{{ confusion_matrix_b64 }}" alt="Confusion Matrix">
  </div>
  {% endif %}

  <div class="section">
    <h2>Recommendations</h2>
    <ul>
      {% for rec in recommendations %}
      <li>{{ rec }}</li>
      {% endfor %}
    </ul>
  </div>

  <footer>Ice Cream Shelf Detector &mdash; Internship Project</footer>
</div>
</body>
</html>"""


def _img_to_b64(path: Path) -> str | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _build_recommendations(report: dict[str, Any]) -> list[str]:
    recs: list[str] = []
    overall = report.get("overall", {})
    if overall.get("mAP50", 0) < 0.7:
        recs.append("mAP@0.5 is below 0.70 — consider collecting more training data or running longer.")
    if overall.get("counting_accuracy", 0) < 0.8:
        recs.append("Counting accuracy is below 80% — review IoU threshold and confidence settings.")
    worst = report.get("worst_classes", [])
    if worst:
        recs.append(f"Focus data collection on: {', '.join(worst)}.")
    if not recs:
        recs.append("Model performance looks good. Consider deploying and monitoring in production.")
    return recs


def generate_html_report(
    eval_report_path: str | Path,
    output_path: str | Path = "reports/evaluation_report.html",
    confusion_matrix_image: str | Path | None = None,
) -> Path:
    """Build a standalone HTML report from an evaluation JSON report.

    Args:
        eval_report_path: Path to the JSON produced by ``evaluator.evaluate_model()``.
        output_path: Where to save the HTML file.
        confusion_matrix_image: Optional path to a confusion matrix PNG.

    Returns:
        Path to the generated HTML file.
    """
    with open(eval_report_path) as f:
        report: dict[str, Any] = json.load(f)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_summary = {
        "Weights": report.get("weights", "unknown"),
        "Evaluation Split": report.get("split", "test"),
    }

    cm_b64 = None
    if confusion_matrix_image:
        cm_b64 = _img_to_b64(Path(confusion_matrix_image))

    template = Template(_HTML_TEMPLATE)
    html = template.render(
        title="Ice Cream Shelf Detector — Evaluation Report",
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        model_summary=model_summary,
        overall=type("O", (), report.get("overall", {}))(),
        per_class=report.get("per_class", {}),
        worst_classes=report.get("worst_classes", []),
        confusion_matrix_b64=cm_b64,
        recommendations=_build_recommendations(report),
    )

    output_path.write_text(html, encoding="utf-8")
    log.info("HTML report → %s", output_path)
    return output_path


if __name__ == "__main__":
    import click

    @click.command()
    @click.argument("eval_report", type=click.Path(exists=True))
    @click.option("--output", default="reports/evaluation_report.html", show_default=True)
    @click.option("--confusion-matrix", default=None, type=click.Path())
    def cli(eval_report, output, confusion_matrix):
        """Generate an HTML evaluation report."""
        path = generate_html_report(eval_report, output, confusion_matrix)
        print(f"Report saved → {path}")

    cli()
