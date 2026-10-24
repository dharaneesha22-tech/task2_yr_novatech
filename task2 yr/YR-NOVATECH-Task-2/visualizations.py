"""
visualizations.py — Reusable Plotly/Seaborn chart builders.

All functions return Plotly figures (or Matplotlib figures for
heatmaps) so they can be unit-tested without a running Streamlit
server. The Streamlit UI calls st.plotly_chart(fig) on the results.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Colour palette consistent with the light theme
PRIMARY_COLOR = "#1f77b4"
ACCENT_COLOR = "#ff7f0e"
SUCCESS_COLOR = "#2ca02c"
WARN_COLOR = "#d62728"

MAX_CATEGORIES = 20  # Limit bar charts to prevent clutter


# ---------------------------------------------------------------------------
# Numerical charts
# ---------------------------------------------------------------------------

def histogram(
    df: pd.DataFrame,
    column: str,
    color_col: Optional[str] = None,
    nbins: int = 30,
    title: Optional[str] = None,
) -> go.Figure:
    """Interactive histogram for a numerical column."""
    title = title or f"Distribution of {column}"
    fig = px.histogram(
        df,
        x=column,
        color=color_col,
        nbins=nbins,
        title=title,
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(bargap=0.05, xaxis_title=column, yaxis_title="Count")
    return fig


def box_plot(
    df: pd.DataFrame,
    column: str,
    color_col: Optional[str] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """Box plot for a numerical column, optionally grouped by a category."""
    title = title or f"Box Plot — {column}"
    fig = px.box(
        df,
        y=column,
        x=color_col,
        color=color_col,
        title=title,
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    return fig


def scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    sample_n: int = 2000,
    title: Optional[str] = None,
) -> go.Figure:
    """Scatter plot. Samples up to sample_n rows to keep it fast."""
    plot_df = df.sample(min(sample_n, len(df)), random_state=42) if len(df) > sample_n else df
    title = title or f"{y_col} vs {x_col}"
    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title,
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set2,
        opacity=0.7,
    )
    return fig


def correlation_heatmap(df: pd.DataFrame, title: str = "Correlation Heatmap") -> go.Figure:
    """Plotly correlation heatmap for all numerical columns."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty or num_df.shape[1] < 2:
        fig = go.Figure()
        fig.update_layout(title="Not enough numerical columns for a correlation heatmap.")
        return fig

    corr = num_df.corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title=title,
        template="plotly_white",
    )
    fig.update_layout(coloraxis_showscale=True)
    return fig


def multi_box_plot(df: pd.DataFrame, columns: list[str], title: str = "Box Plots") -> go.Figure:
    """Overlaid box plots for multiple numerical columns (normalised for scale)."""
    fig = go.Figure()
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        fig.add_trace(go.Box(y=series, name=col, boxmean=True))
    fig.update_layout(title=title, template="plotly_white", yaxis_title="Value")
    return fig


# ---------------------------------------------------------------------------
# Categorical charts
# ---------------------------------------------------------------------------

def bar_chart(
    df: pd.DataFrame,
    column: str,
    top_n: int = MAX_CATEGORIES,
    color_col: Optional[str] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """Horizontal bar chart for a categorical column."""
    vc = df[column].value_counts(dropna=False).head(top_n).reset_index()
    vc.columns = [column, "Count"]
    title = title or f"Top {min(top_n, len(vc))} Values — {column}"
    fig = px.bar(
        vc,
        x="Count",
        y=column,
        orientation="h",
        title=title,
        template="plotly_white",
        color_discrete_sequence=[PRIMARY_COLOR],
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig


def pie_chart(
    df: pd.DataFrame,
    column: str,
    top_n: int = 10,
    title: Optional[str] = None,
) -> go.Figure:
    """Pie/donut chart for a categorical column."""
    vc = df[column].value_counts(dropna=False).head(top_n)
    title = title or f"Distribution — {column}"
    fig = px.pie(
        values=vc.values,
        names=vc.index.astype(str),
        title=title,
        template="plotly_white",
        hole=0.35,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    return fig


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

def missing_values_bar(df: pd.DataFrame, title: str = "Missing Values per Column") -> go.Figure:
    """Bar chart of missing-value counts for each column."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    if missing.empty:
        fig = go.Figure()
        fig.update_layout(title="No missing values found.")
        return fig
    pct = (missing / len(df) * 100).round(2)
    fig = go.Figure(
        go.Bar(
            x=missing.values,
            y=missing.index.tolist(),
            orientation="h",
            text=[f"{p}%" for p in pct.values],
            textposition="outside",
            marker_color=WARN_COLOR,
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Missing Count",
        yaxis_title="Column",
    )
    return fig


def missing_values_heatmap(df: pd.DataFrame, max_rows: int = 200) -> go.Figure:
    """
    Heatmap showing missingness pattern (rows × columns).
    Samples max_rows rows for performance.
    """
    sample = df.sample(min(max_rows, len(df)), random_state=42)
    z = sample.isnull().astype(int)
    fig = px.imshow(
        z,
        color_continuous_scale=[[0, "#e8f5e9"], [1, "#c62828"]],
        title="Missing Value Pattern (red = missing)",
        template="plotly_white",
        aspect="auto",
    )
    fig.update_coloraxes(showscale=False)
    return fig


# ---------------------------------------------------------------------------
# Before / After comparison
# ---------------------------------------------------------------------------

def comparison_bar(
    labels: list[str],
    before_vals: list[float],
    after_vals: list[float],
    title: str = "Before vs After Comparison",
    y_label: str = "Count",
) -> go.Figure:
    """Grouped bar chart comparing before and after values."""
    fig = go.Figure(
        data=[
            go.Bar(name="Before", x=labels, y=before_vals, marker_color=ACCENT_COLOR),
            go.Bar(name="After", x=labels, y=after_vals, marker_color=SUCCESS_COLOR),
        ]
    )
    fig.update_layout(
        barmode="group",
        title=title,
        template="plotly_white",
        yaxis_title=y_label,
    )
    return fig


def distribution_comparison(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
    column: str,
    title: Optional[str] = None,
) -> go.Figure:
    """Overlay two histograms for the same column before and after cleaning."""
    title = title or f"Distribution Comparison — {column}"
    fig = go.Figure()
    for label, df, color in [
        ("Original", original, ACCENT_COLOR),
        ("Cleaned", cleaned, SUCCESS_COLOR),
    ]:
        s = df[column].dropna()
        if s.empty:
            continue
        fig.add_trace(
            go.Histogram(
                x=s,
                name=label,
                opacity=0.65,
                marker_color=color,
                nbinsx=30,
            )
        )
    fig.update_layout(
        barmode="overlay",
        title=title,
        template="plotly_white",
        xaxis_title=column,
        yaxis_title="Count",
    )
    return fig


# ---------------------------------------------------------------------------
# EDA summary
# ---------------------------------------------------------------------------

def grouped_bar(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: Optional[str] = None,
    agg: str = "mean",
    title: Optional[str] = None,
) -> go.Figure:
    """
    Bar chart with optional grouping. y_col is aggregated by agg function.
    """
    title = title or f"{agg.capitalize()} of {y_col} by {x_col}"
    if group_col and group_col in df.columns:
        grouped = (
            df.groupby([x_col, group_col])[y_col]
            .agg(agg)
            .reset_index()
        )
        fig = px.bar(
            grouped,
            x=x_col,
            y=y_col,
            color=group_col,
            barmode="group",
            title=title,
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
    else:
        grouped = df.groupby(x_col)[y_col].agg(agg).reset_index()
        fig = px.bar(
            grouped,
            x=x_col,
            y=y_col,
            title=title,
            template="plotly_white",
            color_discrete_sequence=[PRIMARY_COLOR],
        )
    return fig


def workflow_progress_chart(steps: list[str], completed: list[bool]) -> go.Figure:
    """Horizontal step indicator showing workflow progress."""
    colors = [SUCCESS_COLOR if done else "#b0bec5" for done in completed]
    fig = go.Figure(
        go.Bar(
            x=[1] * len(steps),
            y=steps,
            orientation="h",
            marker_color=colors,
            text=["✔" if done else "○" for done in completed],
            textposition="inside",
        )
    )
    fig.update_layout(
        template="plotly_white",
        xaxis={"visible": False},
        yaxis={"autorange": "reversed"},
        height=280,
        margin={"l": 20, "r": 20, "t": 10, "b": 10},
        showlegend=False,
    )
    return fig
