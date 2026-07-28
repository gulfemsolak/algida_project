"""Reusable Plotly chart components, themed to the IceVision dark + neon frost palette."""
from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.theme import (
    ICE,
    MINT,
    PLOTLY_THEME,
    format_class_name,
)


def _themed(fig: go.Figure, height: int | None = None, **overrides) -> go.Figure:
    """Apply the shared dark + neon theme layout to any Plotly figure."""
    fig.update_layout(**PLOTLY_THEME)
    if height:
        fig.update_layout(height=height)
    if overrides:
        fig.update_layout(**overrides)
    return fig


def category_count_bar(summary: dict[str, dict[str, Any]]) -> go.Figure:
    """Horizontal bar chart of detected product counts per category."""
    categories = list(summary.keys())
    counts = [v["count"] for v in summary.values()]
    confs = [round(v.get("avg_confidence", 0) * 100, 1) for v in summary.values()]
    labels = [format_class_name(c) for c in categories]

    df = pd.DataFrame({"Ürün": labels, "Adet": counts, "Ort. Güven %": confs})
    df.sort_values("Adet", ascending=True, inplace=True)

    fig = px.bar(
        df,
        x="Adet",
        y="Ürün",
        orientation="h",
        color="Ort. Güven %",
        color_continuous_scale=[ICE, MINT],
        text="Adet",
        title="Ürün bazında tespit adedi",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title="Adet",
        yaxis_title="",
        coloraxis_colorbar_title="Ort. Güven %",
        height=max(300, len(categories) * 38),
    )
    return _themed(fig)


def fill_rate_histogram(df: pd.DataFrame) -> go.Figure:
    """Distribution of fill rates across a batch of processed images."""
    fig = px.histogram(
        df,
        x="fill_rate",
        title="Doluluk oranı dağılımı",
        color_discrete_sequence=[ICE],
        range_x=[-0.05, 1.05],
    )
    fig.update_traces(xbins=dict(start=0.0, end=1.0, size=0.1))
    fig.update_layout(
        xaxis=dict(tickformat=".0%", range=[-0.05, 1.05], title="Doluluk oranı"),
        yaxis=dict(dtick=1, title="Görsel sayısı"),
    )
    return _themed(fig)
