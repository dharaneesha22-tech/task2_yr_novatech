"""
tests/test_data_quality.py

Unit tests for data_quality.py using small synthetic DataFrames only.
"""

import pytest
import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_quality as dq


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [25, 30, np.nan, 40, 45, 30, 25],
        "salary": [50000, 60000, 70000, np.nan, 80000, 60000, 50000],
        "city": ["New York", "London", None, "Paris", "Tokyo", "London", "New York"],
        "score": [8.5, 7.0, 9.0, 6.5, 8.0, 7.0, 8.5],
        "constant_col": [1, 1, 1, 1, 1, 1, 1],
    })


@pytest.fixture
def clean_df():
    return pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [10.0, 20.0, 30.0, 40.0, 50.0],
        "label": ["a", "b", "c", "a", "b"],
    })


# ---------------------------------------------------------------------------
# generate_quality_report tests
# ---------------------------------------------------------------------------

def test_report_rows_columns(sample_df):
    report = dq.generate_quality_report(sample_df)
    assert report["rows"] == 7
    assert report["columns"] == 5


def test_report_missing_total(sample_df):
    report = dq.generate_quality_report(sample_df)
    # age has 1 NaN, salary has 1 NaN, city has 1 NaN → 3 total
    assert report["missing_total"] == 3


def test_report_missing_per_column(sample_df):
    report = dq.generate_quality_report(sample_df)
    assert report["missing_per_column"]["age"] == 1
    assert report["missing_per_column"]["salary"] == 1
    assert report["missing_per_column"]["city"] == 1


def test_report_duplicate_rows(sample_df):
    report = dq.generate_quality_report(sample_df)
    # Rows 0 and 6 are identical, rows 1 and 5 are identical → 2 duplicates
    assert report["duplicate_rows"] == 2


def test_report_completeness(clean_df):
    report = dq.generate_quality_report(clean_df)
    assert report["completeness_pct"] == 100.0


def test_report_completeness_partial(sample_df):
    report = dq.generate_quality_report(sample_df)
    total_cells = 7 * 5
    expected = round((total_cells - 3) / total_cells * 100, 2)
    assert report["completeness_pct"] == expected


def test_report_constant_cols(sample_df):
    report = dq.generate_quality_report(sample_df)
    assert "constant_col" in report["constant_cols"]


def test_report_numerical_cols(clean_df):
    report = dq.generate_quality_report(clean_df)
    assert "x" in report["numerical_cols"]
    assert "y" in report["numerical_cols"]


def test_report_categorical_cols(clean_df):
    report = dq.generate_quality_report(clean_df)
    assert "label" in report["categorical_cols"]


def test_report_numerical_summary(clean_df):
    report = dq.generate_quality_report(clean_df)
    assert not report["numerical_summary"].empty
    assert "mean" in report["numerical_summary"].columns


def test_report_categorical_summary(clean_df):
    report = dq.generate_quality_report(clean_df)
    assert "label" in report["categorical_summary"]
    label_summary = report["categorical_summary"]["label"]
    assert "Value" in label_summary.columns
    assert "Count" in label_summary.columns


def test_report_empty_df():
    report = dq.generate_quality_report(pd.DataFrame())
    assert report["rows"] == 0
    assert report["columns"] == 0
    assert report["completeness_pct"] == 100.0


def test_report_single_row():
    df = pd.DataFrame({"a": [1], "b": ["hello"]})
    report = dq.generate_quality_report(df)
    assert report["rows"] == 1
    assert report["missing_total"] == 0


def test_report_all_missing():
    df = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
    report = dq.generate_quality_report(df)
    assert report["missing_total"] == 4
    assert report["completeness_pct"] == 0.0


def test_report_dtypes(sample_df):
    report = dq.generate_quality_report(sample_df)
    assert "age" in report["dtypes"]
    assert "city" in report["dtypes"]


def test_report_unique_per_column(clean_df):
    report = dq.generate_quality_report(clean_df)
    assert report["unique_per_column"]["label"] == 3


# ---------------------------------------------------------------------------
# missingness_table tests
# ---------------------------------------------------------------------------

def test_missingness_table_has_columns(sample_df):
    result = dq.missingness_table(sample_df)
    assert "Column" in result.columns
    assert "Missing Count" in result.columns
    assert "Missing %" in result.columns


def test_missingness_table_sorted(sample_df):
    result = dq.missingness_table(sample_df)
    counts = result["Missing Count"].tolist()
    assert counts == sorted(counts, reverse=True)


def test_missingness_table_no_missing(clean_df):
    result = dq.missingness_table(clean_df)
    assert result["Missing Count"].sum() == 0


def test_missingness_table_empty_df():
    result = dq.missingness_table(pd.DataFrame())
    assert result.empty


# ---------------------------------------------------------------------------
# column_quality_table tests
# ---------------------------------------------------------------------------

def test_column_quality_table_shape(sample_df):
    result = dq.column_quality_table(sample_df)
    assert len(result) == sample_df.shape[1]


def test_column_quality_table_columns(sample_df):
    result = dq.column_quality_table(sample_df)
    expected_cols = ["Column", "Data Type", "Non-Null Count", "Missing Count", "Missing %", "Unique Values", "Is Constant"]
    for c in expected_cols:
        assert c in result.columns


def test_column_quality_table_constant_flag(sample_df):
    result = dq.column_quality_table(sample_df)
    constant_row = result[result["Column"] == "constant_col"]
    assert constant_row["Is Constant"].iloc[0] == True


def test_column_quality_table_empty():
    result = dq.column_quality_table(pd.DataFrame())
    assert result.empty
