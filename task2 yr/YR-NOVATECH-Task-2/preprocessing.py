"""
preprocessing.py — Data cleaning and feature engineering engine.

All functions are pure transformations: they receive a DataFrame and
parameters, and return a new DataFrame. They never modify in place and
never touch session state — that is handled by app.py.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd


# ===========================================================================
# A. Missing-value handling
# ===========================================================================

def identify_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary DataFrame of missing values per column.
    """
    if df.empty:
        return pd.DataFrame(columns=["column", "missing_count", "missing_pct"])
    rows = len(df)
    missing = df.isnull().sum()
    pct = (missing / rows * 100).round(2)
    result = pd.DataFrame({"column": df.columns, "missing_count": missing.values, "missing_pct": pct.values})
    return result[result["missing_count"] > 0].sort_values("missing_count", ascending=False).reset_index(drop=True)


def impute_numerical(
    df: pd.DataFrame,
    columns: list[str],
    strategy: str = "mean",
    fill_value: Optional[float] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Impute missing values in numerical columns.

    Parameters
    ----------
    columns  : columns to impute
    strategy : 'mean', 'median', or 'constant'
    fill_value : used when strategy='constant'

    Returns
    -------
    (new_df, stats_dict)
    """
    df = df.copy()
    stats: dict = {}

    for col in columns:
        if col not in df.columns:
            continue
        before = int(df[col].isnull().sum())
        if before == 0:
            continue
        if strategy == "mean":
            value = df[col].mean()
        elif strategy == "median":
            value = df[col].median()
        elif strategy == "constant":
            if fill_value is None:
                raise ValueError("fill_value must be provided when strategy='constant'.")
            value = fill_value
        else:
            raise ValueError(f"Unknown strategy: '{strategy}'.")

        df[col] = df[col].fillna(value)
        after = int(df[col].isnull().sum())
        stats[col] = {"strategy": strategy, "value_used": value, "filled": before - after}

    return df, stats


def impute_categorical(
    df: pd.DataFrame,
    columns: list[str],
    strategy: str = "mode",
    fill_value: Optional[str] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Impute missing values in categorical columns.

    Parameters
    ----------
    columns  : columns to impute
    strategy : 'mode' or 'constant'
    fill_value : used when strategy='constant'
    """
    df = df.copy()
    stats: dict = {}

    for col in columns:
        if col not in df.columns:
            continue
        before = int(df[col].isnull().sum())
        if before == 0:
            continue
        if strategy == "mode":
            mode_vals = df[col].mode()
            value = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"
        elif strategy == "constant":
            if fill_value is None:
                raise ValueError("fill_value must be provided when strategy='constant'.")
            value = fill_value
        else:
            raise ValueError(f"Unknown strategy: '{strategy}'.")

        df[col] = df[col].fillna(value)
        after = int(df[col].isnull().sum())
        stats[col] = {"strategy": strategy, "value_used": value, "filled": before - after}

    return df, stats


def drop_rows_with_missing(
    df: pd.DataFrame, columns: Optional[list[str]] = None
) -> tuple[pd.DataFrame, int]:
    """
    Drop rows that have any missing value in the specified columns.
    If columns is None, consider all columns.

    Returns (new_df, rows_dropped).
    """
    before = len(df)
    if columns:
        df = df.dropna(subset=columns).reset_index(drop=True)
    else:
        df = df.dropna().reset_index(drop=True)
    return df, before - len(df)


def drop_columns_with_missing(
    df: pd.DataFrame, threshold_pct: float = 50.0
) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop columns whose missing percentage exceeds threshold_pct.
    Only drops when explicitly called; never automatic.

    Returns (new_df, dropped_col_names).
    """
    rows = len(df)
    if rows == 0:
        return df, []
    dropped = [
        c for c in df.columns if df[c].isnull().sum() / rows * 100 >= threshold_pct
    ]
    return df.drop(columns=dropped), dropped


# ===========================================================================
# B. Duplicate handling
# ===========================================================================

def count_duplicates(df: pd.DataFrame) -> int:
    """Return the number of exact duplicate rows."""
    return int(df.duplicated().sum())


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove exact duplicate rows, keeping the first occurrence.

    Returns (new_df, rows_removed).
    """
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, before - len(df)


# ===========================================================================
# C. Data type correction
# ===========================================================================

def convert_to_numeric(
    df: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, dict]:
    """
    Attempt to convert selected columns to numeric (float64).

    Returns (new_df, result_dict) where result_dict maps column name to
    {'converted': True/False, 'coerced_nulls': count, 'error': str_or_None}.
    """
    df = df.copy()
    results: dict = {}
    for col in columns:
        if col not in df.columns:
            results[col] = {"converted": False, "coerced_nulls": 0, "error": "Column not found."}
            continue
        before_nulls = int(df[col].isnull().sum())
        converted = pd.to_numeric(df[col], errors="coerce")
        after_nulls = int(converted.isnull().sum())
        new_nulls = after_nulls - before_nulls
        df[col] = converted
        results[col] = {
            "converted": True,
            "coerced_nulls": new_nulls,
            "error": None,
        }
    return df, results


def parse_datetime_column(
    df: pd.DataFrame, column: str, fmt: Optional[str] = None
) -> tuple[pd.DataFrame, int, str]:
    """
    Parse a column as datetime.

    Returns (new_df, error_count, message).
    """
    if column not in df.columns:
        return df, 0, f"Column '{column}' not found."

    df = df.copy()
    before_nulls = int(df[column].isnull().sum())
    try:
        if fmt:
            df[column] = pd.to_datetime(df[column], format=fmt, errors="coerce")
        else:
            df[column] = pd.to_datetime(df[column], errors="coerce")
        after_nulls = int(df[column].isnull().sum())
        errors = after_nulls - before_nulls
        return df, errors, f"Parsed '{column}' as datetime. {errors} value(s) could not be converted (set to NaT)."
    except Exception as exc:  # pylint: disable=broad-except
        return df, 0, f"Failed to parse '{column}': {exc}"


# ===========================================================================
# D. Text normalisation
# ===========================================================================

def trim_whitespace(df: pd.DataFrame, columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Strip leading/trailing whitespace from string columns."""
    df = df.copy()
    target = columns if columns else df.select_dtypes(include="object").columns.tolist()
    for col in target:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.strip()
    return df


def normalize_whitespace(df: pd.DataFrame, columns: Optional[list[str]] = None) -> pd.DataFrame:
    """Collapse repeated internal whitespace to a single space."""
    df = df.copy()
    target = columns if columns else df.select_dtypes(include="object").columns.tolist()
    for col in target:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.replace(r"\s+", " ", regex=True).str.strip()
    return df


def standardize_casing(
    df: pd.DataFrame,
    columns: list[str],
    case: str = "lower",
) -> pd.DataFrame:
    """
    Standardise string casing for the given columns.

    Parameters
    ----------
    case : 'lower', 'upper', or 'title'
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns or df[col].dtype != object:
            continue
        if case == "lower":
            df[col] = df[col].str.lower()
        elif case == "upper":
            df[col] = df[col].str.upper()
        elif case == "title":
            df[col] = df[col].str.title()
    return df


# ===========================================================================
# E. Outlier detection and handling
# ===========================================================================

def detect_outliers_iqr(
    df: pd.DataFrame, columns: list[str], multiplier: float = 1.5
) -> dict[str, dict]:
    """
    Detect outliers using the IQR method.

    Returns a dict keyed by column name:
      { 'q1', 'q3', 'iqr', 'lower', 'upper', 'outlier_count', 'outlier_pct' }
    """
    results: dict = {}
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty or not pd.api.types.is_numeric_dtype(series):
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        count = int(mask.sum())
        results[col] = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower": lower,
            "upper": upper,
            "outlier_count": count,
            "outlier_pct": round(count / len(df) * 100, 2) if len(df) else 0.0,
        }
    return results


def handle_outliers(
    df: pd.DataFrame,
    columns: list[str],
    method: str = "cap",
    multiplier: float = 1.5,
) -> tuple[pd.DataFrame, dict]:
    """
    Handle outliers by capping or removing them.

    Parameters
    ----------
    method : 'cap' (winsorise) or 'remove'

    Returns (new_df, stats_dict).
    """
    df = df.copy()
    stats: dict = {}
    outlier_info = detect_outliers_iqr(df, columns, multiplier)

    for col, info in outlier_info.items():
        lower, upper = info["lower"], info["upper"]
        before_rows = len(df)
        before_count = info["outlier_count"]

        if method == "cap":
            df[col] = df[col].clip(lower=lower, upper=upper)
            stats[col] = {
                "method": "cap",
                "lower": lower,
                "upper": upper,
                "values_capped": before_count,
            }
        elif method == "remove":
            mask = (df[col] >= lower) & (df[col] <= upper)
            df = df[mask | df[col].isnull()].reset_index(drop=True)
            removed = before_rows - len(df)
            stats[col] = {
                "method": "remove",
                "lower": lower,
                "upper": upper,
                "rows_removed": removed,
            }
    return df, stats


# ===========================================================================
# F. Column operations
# ===========================================================================

def drop_columns(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop specified columns. Only drops columns that actually exist.

    Returns (new_df, actually_dropped_list).
    """
    existing = [c for c in columns if c in df.columns]
    return df.drop(columns=existing), existing


def rename_column(df: pd.DataFrame, old_name: str, new_name: str) -> tuple[pd.DataFrame, str]:
    """
    Rename a single column.

    Returns (new_df, message).
    """
    if old_name not in df.columns:
        return df, f"Column '{old_name}' not found."
    if new_name in df.columns:
        return df, f"Column '{new_name}' already exists. Choose a different name."
    if not new_name.strip():
        return df, "New column name cannot be empty."
    df = df.rename(columns={old_name: new_name})
    return df, f"Renamed '{old_name}' → '{new_name}'."


# ===========================================================================
# G. Feature engineering
# ===========================================================================

ALLOWED_ARITHMETIC_OPS: dict[str, str] = {
    "add (+)": "+",
    "subtract (-)": "-",
    "multiply (*)": "*",
    "divide (/)": "/",
}


def create_arithmetic_feature(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    operation: str,
    new_col_name: str,
    drop_source: bool = False,
) -> tuple[pd.DataFrame, str]:
    """
    Create a new column from two existing numerical columns using a safe
    arithmetic operation. No eval() is used.

    Parameters
    ----------
    operation : one of '+', '-', '*', '/'
    """
    if col_a not in df.columns:
        return df, f"Column '{col_a}' not found."
    if col_b not in df.columns:
        return df, f"Column '{col_b}' not found."
    if new_col_name in df.columns:
        return df, f"Column '{new_col_name}' already exists. Choose a different name."
    if not new_col_name.strip():
        return df, "New feature name cannot be empty."
    if operation not in ("+", "-", "*", "/"):
        return df, f"Unsupported operation '{operation}'. Use +, -, *, or /."

    df = df.copy()
    a = pd.to_numeric(df[col_a], errors="coerce")
    b = pd.to_numeric(df[col_b], errors="coerce")

    if operation == "+":
        df[new_col_name] = a + b
    elif operation == "-":
        df[new_col_name] = a - b
    elif operation == "*":
        df[new_col_name] = a * b
    elif operation == "/":
        # Guard against division by zero — result becomes NaN where b==0
        safe_b = b.replace(0, np.nan)
        df[new_col_name] = a / safe_b

    if drop_source:
        df = df.drop(columns=[col_a, col_b], errors="ignore")

    return df, f"Created '{new_col_name}' = {col_a} {operation} {col_b}."


def extract_datetime_features(
    df: pd.DataFrame,
    column: str,
    features: list[str],
    drop_source: bool = False,
) -> tuple[pd.DataFrame, str]:
    """
    Extract date-part features from a datetime column.

    features : subset of ['year', 'month', 'day', 'dayofweek', 'hour', 'quarter']
    """
    if column not in df.columns:
        return df, f"Column '{column}' not found."

    df = df.copy()
    col_dt = pd.to_datetime(df[column], errors="coerce")
    extracted: list[str] = []

    feature_map = {
        "year": col_dt.dt.year,
        "month": col_dt.dt.month,
        "day": col_dt.dt.day,
        "dayofweek": col_dt.dt.dayofweek,
        "hour": col_dt.dt.hour,
        "quarter": col_dt.dt.quarter,
    }
    for feat in features:
        if feat not in feature_map:
            continue
        new_col = f"{column}_{feat}"
        if new_col in df.columns:
            new_col = f"{new_col}_extracted"
        df[new_col] = feature_map[feat]
        extracted.append(new_col)

    if drop_source:
        df = df.drop(columns=[column], errors="ignore")

    msg = f"Extracted {extracted} from '{column}'." if extracted else "No features extracted."
    return df, msg


def create_bins(
    df: pd.DataFrame,
    column: str,
    bins: int | list[float],
    labels: Optional[list[str]],
    new_col_name: str,
    drop_source: bool = False,
) -> tuple[pd.DataFrame, str]:
    """
    Bin a numerical column into categories.

    Parameters
    ----------
    bins   : number of equal-width bins (int) or explicit bin edges (list)
    labels : list of string labels matching the number of intervals
    """
    if column not in df.columns:
        return df, f"Column '{column}' not found."
    if new_col_name in df.columns:
        return df, f"Column '{new_col_name}' already exists."
    if not new_col_name.strip():
        return df, "New column name cannot be empty."

    df = df.copy()
    try:
        df[new_col_name] = pd.cut(
            df[column],
            bins=bins,
            labels=labels,
            include_lowest=True,
        )
    except Exception as exc:  # pylint: disable=broad-except
        return df, f"Binning failed: {exc}"

    if drop_source:
        df = df.drop(columns=[column], errors="ignore")

    return df, f"Created '{new_col_name}' from '{column}' with {bins if isinstance(bins, int) else len(bins)-1} bins."


def create_interaction_feature(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    new_col_name: str,
    drop_source: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Create a multiplicative interaction feature col_a * col_b."""
    return create_arithmetic_feature(df, col_a, col_b, "*", new_col_name, drop_source)
