"""
utils.py — Shared utility helpers for the YR NOVATECH Data Platform.

Provides session-state management, file validation, safe download of
the bundled public dataset, and general-purpose helpers used by other
modules and app.py.
"""

from __future__ import annotations

import io
import os
import json
import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_UPLOAD_MB: int = 50
MAX_UPLOAD_BYTES: int = MAX_UPLOAD_MB * 1024 * 1024

# Public dataset — UCI Adult Income
# Source: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
DATASET_URL: str = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)
DATASET_COLUMNS: list[str] = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    """Initialise all session-state keys so every page can rely on them."""
    defaults: dict = {
        "original_df": None,         # untouched upload copy
        "working_df": None,          # current cleaned/engineered copy
        "preprocessing_log": [],     # list of operation dicts
        "dataset_name": "",
        "dataset_source": "",
        "upload_validated": False,
        "operations_count": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session() -> None:
    """Clear all dataset-related session state."""
    keys_to_clear = [
        "original_df",
        "working_df",
        "preprocessing_log",
        "dataset_name",
        "dataset_source",
        "upload_validated",
        "operations_count",
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    init_session_state()


def log_operation(
    operation: str,
    columns: list[str],
    params: dict,
    rows_before: int,
    rows_after: int,
    summary: str,
) -> None:
    """Append a preprocessing operation to the session log."""
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "operation": operation,
        "columns": ", ".join(columns) if columns else "—",
        "parameters": json.dumps(params),
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_changed": rows_before - rows_after,
        "summary": summary,
    }
    st.session_state.preprocessing_log.append(entry)
    st.session_state.operations_count = len(st.session_state.preprocessing_log)


def get_log_df() -> pd.DataFrame:
    """Return the preprocessing log as a DataFrame."""
    if not st.session_state.preprocessing_log:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "operation",
                "columns",
                "parameters",
                "rows_before",
                "rows_after",
                "rows_changed",
                "summary",
            ]
        )
    return pd.DataFrame(st.session_state.preprocessing_log)


# ---------------------------------------------------------------------------
# File validation
# ---------------------------------------------------------------------------

def validate_uploaded_file(uploaded_file) -> tuple[bool, str]:
    """
    Validate an uploaded file object from st.file_uploader.

    Returns
    -------
    (is_valid, message)
    """
    if uploaded_file is None:
        return False, "No file provided."

    name: str = uploaded_file.name.lower()
    if not (name.endswith(".csv") or name.endswith(".xlsx") or name.endswith(".xls")):
        return False, f"Unsupported file type: '{uploaded_file.name}'. Please upload a .csv or .xlsx file."

    size: int = uploaded_file.size
    if size == 0:
        return False, "The uploaded file is empty (0 bytes)."
    if size > MAX_UPLOAD_BYTES:
        mb = size / (1024 * 1024)
        return False, (
            f"File is {mb:.1f} MB, which exceeds the {MAX_UPLOAD_MB} MB limit. "
            "Please upload a smaller file."
        )
    return True, "File is valid."


def load_uploaded_file(uploaded_file) -> tuple[Optional[pd.DataFrame], str]:
    """
    Parse an uploaded CSV or Excel file into a DataFrame.

    Returns
    -------
    (dataframe_or_None, message)
    """
    valid, msg = validate_uploaded_file(uploaded_file)
    if not valid:
        return None, msg

    name: str = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            # Try UTF-8 first, fall back to latin-1
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="latin-1")
        else:
            try:
                import openpyxl  # noqa: F401
                df = pd.read_excel(uploaded_file)
            except ImportError:
                return None, (
                    "Excel support requires 'openpyxl'. "
                    "Install it with: pip install openpyxl"
                )

        if df.empty:
            return None, "The file was parsed but contains no rows."
        if df.shape[1] == 0:
            return None, "The file has no columns."

        return df, f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns."
    except pd.errors.ParserError as exc:
        return None, f"CSV parsing error: {exc}"
    except Exception as exc:  # pylint: disable=broad-except
        return None, f"Could not read file: {exc}"


# ---------------------------------------------------------------------------
# Public dataset download
# ---------------------------------------------------------------------------

def load_public_dataset() -> tuple[Optional[pd.DataFrame], str]:
    """
    Attempt to download the UCI Adult Income dataset.

    Returns
    -------
    (dataframe_or_None, status_message)
    """
    try:
        df = pd.read_csv(
            DATASET_URL,
            header=None,
            names=DATASET_COLUMNS,
            na_values=" ?",
            skipinitialspace=True,
        )
        rows, cols = df.shape
        if rows < 100:
            return None, (
                f"Downloaded data has only {rows} rows — "
                "the remote file may have changed."
            )
        return df, (
            f"UCI Adult Income dataset loaded: {rows:,} rows × {cols} columns."
        )
    except Exception as exc:  # pylint: disable=broad-except
        return None, (
            f"Could not download the public dataset: {exc}. "
            "Please upload your own CSV file."
        )


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def safe_filename(name: str) -> str:
    """Return a filesystem-safe version of name."""
    keep = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")
    return "".join(c if c in keep else "_" for c in name)


def df_memory_usage(df: pd.DataFrame) -> str:
    """Return a human-readable memory usage string for df."""
    bytes_used = df.memory_usage(deep=True).sum()
    if bytes_used < 1024:
        return f"{bytes_used} B"
    if bytes_used < 1024 ** 2:
        return f"{bytes_used / 1024:.1f} KB"
    return f"{bytes_used / (1024 ** 2):.2f} MB"


def completeness_pct(df: pd.DataFrame) -> float:
    """Return the percentage of non-null cells in df, or 100.0 if empty."""
    if df.empty or df.size == 0:
        return 100.0
    total = df.size
    missing = int(df.isnull().sum().sum())
    return round((total - missing) / total * 100, 2)


def numerical_cols(df: pd.DataFrame) -> list[str]:
    """Return list of numerical column names."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def categorical_cols(df: pd.DataFrame) -> list[str]:
    """Return list of non-numerical (object/category) column names."""
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def date_cols(df: pd.DataFrame) -> list[str]:
    """Return list of datetime column names."""
    return df.select_dtypes(include=["datetime64"]).columns.tolist()


def build_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a data dictionary DataFrame summarising each column.
    """
    records = []
    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        missing = int(series.isnull().sum())
        pct_missing = round(missing / len(df) * 100, 2) if len(df) > 0 else 0.0
        unique = int(series.nunique(dropna=True))
        sample_vals = series.dropna().head(3).tolist()
        records.append(
            {
                "Column": col,
                "Data Type": dtype,
                "Missing Count": missing,
                "Missing %": pct_missing,
                "Unique Values": unique,
                "Sample Values": ", ".join(str(v) for v in sample_vals),
            }
        )
    return pd.DataFrame(records)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialise df to UTF-8 CSV bytes for st.download_button."""
    return df.to_csv(index=False).encode("utf-8")


def build_pipeline_json(df: pd.DataFrame, log: list[dict]) -> str:
    """Build a JSON string summarising pipeline metadata."""
    summary = {
        "generated_at": datetime.now().isoformat(),
        "dataset_rows": int(df.shape[0]),
        "dataset_columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "operations_count": len(log),
        "operations": log,
    }
    return json.dumps(summary, indent=2, default=str)


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Serialise df to Excel (.xlsx) bytes for st.download_button."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    return buf.getvalue()


def df_to_excel_multi_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    """
    Write multiple DataFrames to separate sheets in one Excel workbook.

    Parameters
    ----------
    sheets : dict mapping sheet name → DataFrame
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            safe_name = sheet_name[:31]  # Excel sheet name limit
            df.to_excel(writer, index=False, sheet_name=safe_name)
    return buf.getvalue()
