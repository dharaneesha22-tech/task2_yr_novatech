"""
data_quality.py — Data quality assessment module for YR NOVATECH Task 2.

All functions accept a pandas DataFrame and return plain Python/pandas
structures so they can be used from both the Streamlit UI and pytest tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core quality report
# ---------------------------------------------------------------------------

def generate_quality_report(df: pd.DataFrame) -> dict[str, Any]:
    """
    Generate a comprehensive data quality report.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyse.

    Returns
    -------
    dict with keys:
        rows, columns, missing_total, missing_per_column, missing_pct_per_column,
        duplicate_rows, unique_per_column, dtypes, numerical_cols,
        categorical_cols, constant_cols, high_cardinality_cols,
        completeness_pct, numerical_summary, categorical_summary,
        suspicious_flags
    """
    if df is None or df.empty:
        return _empty_report()

    rows, cols = df.shape
    total_cells = rows * cols if rows > 0 and cols > 0 else 1

    # Missing values
    missing_series = df.isnull().sum()
    missing_pct_series = (missing_series / rows * 100).round(2)

    # Duplicates
    dup_count = int(df.duplicated().sum())

    # Unique values
    unique_series = df.nunique(dropna=True)

    # Column type classification
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols_list = df.select_dtypes(include=["datetime64"]).columns.tolist()

    # Constant columns (only one unique value, ignoring NaN)
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]

    # High-cardinality categorical columns (>50% unique values)
    high_card_cols = [
        c
        for c in cat_cols
        if rows > 0 and unique_series[c] / rows > 0.5 and unique_series[c] > 20
    ]

    # Completeness
    missing_total = int(missing_series.sum())
    completeness = round((total_cells - missing_total) / total_cells * 100, 2)

    # Numerical summary
    num_summary = (
        df[num_cols].describe().T.round(4) if num_cols else pd.DataFrame()
    )

    # Categorical summaries (top 10 values per column)
    cat_summary: dict[str, pd.DataFrame] = {}
    for c in cat_cols:
        vc = df[c].value_counts(dropna=False).head(10).reset_index()
        vc.columns = ["Value", "Count"]
        vc["Percentage"] = (vc["Count"] / rows * 100).round(2)
        cat_summary[c] = vc

    # Suspicious value flags
    suspicious = _detect_suspicious(df, num_cols)

    return {
        "rows": rows,
        "columns": cols,
        "missing_total": missing_total,
        "missing_per_column": missing_series.to_dict(),
        "missing_pct_per_column": missing_pct_series.to_dict(),
        "duplicate_rows": dup_count,
        "unique_per_column": unique_series.to_dict(),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "numerical_cols": num_cols,
        "categorical_cols": cat_cols,
        "date_cols": date_cols_list,
        "constant_cols": constant_cols,
        "high_cardinality_cols": high_card_cols,
        "completeness_pct": completeness,
        "numerical_summary": num_summary,
        "categorical_summary": cat_summary,
        "suspicious_flags": suspicious,
    }


def _empty_report() -> dict[str, Any]:
    return {
        "rows": 0,
        "columns": 0,
        "missing_total": 0,
        "missing_per_column": {},
        "missing_pct_per_column": {},
        "duplicate_rows": 0,
        "unique_per_column": {},
        "dtypes": {},
        "numerical_cols": [],
        "categorical_cols": [],
        "date_cols": [],
        "constant_cols": [],
        "high_cardinality_cols": [],
        "completeness_pct": 100.0,
        "numerical_summary": pd.DataFrame(),
        "categorical_summary": {},
        "suspicious_flags": {},
    }


def _detect_suspicious(df: pd.DataFrame, num_cols: list[str]) -> dict[str, list[str]]:
    """
    Heuristically flag potentially suspicious values.

    Note: Automated detection is imperfect and requires human review.
    """
    flags: dict[str, list[str]] = {}
    for col in num_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        col_flags: list[str] = []

        # Extreme negative values in columns that are likely non-negative
        if series.min() < 0 and col.lower() not in ("temperature", "profit", "change"):
            if abs(series.min()) > series.abs().mean() * 10:
                col_flags.append(
                    f"Very large negative value detected: min={series.min():.2f}"
                )

        # Values that look like sentinel codes (e.g. 9999, -1, 99)
        for sentinel in [-1, 0, 9999, 99999]:
            pct = (series == sentinel).sum() / len(series) * 100
            if pct > 5:
                col_flags.append(
                    f"Value {sentinel} appears in {pct:.1f}% of rows — "
                    "may be a sentinel/missing code."
                )

        if col_flags:
            flags[col] = col_flags
    return flags


# ---------------------------------------------------------------------------
# Column-level helpers
# ---------------------------------------------------------------------------

def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted DataFrame of missing-value counts and percentages."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Column", "Missing Count", "Missing %"])
    rows = len(df)
    missing = df.isnull().sum()
    pct = (missing / rows * 100).round(2)
    result = pd.DataFrame(
        {"Column": df.columns, "Missing Count": missing.values, "Missing %": pct.values}
    )
    return result.sort_values("Missing Count", ascending=False).reset_index(drop=True)


def column_quality_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column quality table."""
    if df is None or df.empty:
        return pd.DataFrame()
    rows = len(df)
    records = []
    for col in df.columns:
        s = df[col]
        missing = int(s.isnull().sum())
        records.append(
            {
                "Column": col,
                "Data Type": str(s.dtype),
                "Non-Null Count": rows - missing,
                "Missing Count": missing,
                "Missing %": round(missing / rows * 100, 2) if rows else 0,
                "Unique Values": int(s.nunique(dropna=True)),
                "Is Constant": int(s.nunique(dropna=True)) <= 1,
            }
        )
    return pd.DataFrame(records)
