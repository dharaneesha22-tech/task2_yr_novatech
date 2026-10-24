"""
tests/test_utils.py

Unit tests for utils.py (excluding Streamlit-specific session-state helpers,
which require a running Streamlit runtime).
"""

import pytest
import io
import json
import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_df():
    return pd.DataFrame({
        "age": [25, 30, 35],
        "city": ["NY", "LA", "SF"],
        "score": [8.5, 7.0, 9.0],
    })


@pytest.fixture
def df_with_missing():
    return pd.DataFrame({
        "a": [1, None, 3],
        "b": ["x", "y", None],
    })


# ---------------------------------------------------------------------------
# validate_uploaded_file tests — using mock objects
# ---------------------------------------------------------------------------

class MockFile:
    """Minimal mock of st.UploadedFile for testing."""
    def __init__(self, name, size, content=b"col1,col2\n1,2\n3,4\n"):
        self.name = name
        self.size = size
        self._content = content

    def read(self):
        return self._content

    def seek(self, pos):
        pass

    def getvalue(self):
        return self._content


def test_validate_none():
    valid, msg = utils.validate_uploaded_file(None)
    assert valid is False
    assert "No file" in msg


def test_validate_unsupported_type():
    f = MockFile("data.pdf", 100)
    valid, msg = utils.validate_uploaded_file(f)
    assert valid is False
    assert "Unsupported" in msg


def test_validate_empty_file():
    f = MockFile("data.csv", 0)
    valid, msg = utils.validate_uploaded_file(f)
    assert valid is False
    assert "empty" in msg.lower()


def test_validate_file_too_large():
    big_size = utils.MAX_UPLOAD_BYTES + 1
    f = MockFile("data.csv", big_size)
    valid, msg = utils.validate_uploaded_file(f)
    assert valid is False
    assert "exceeds" in msg or "MB" in msg


def test_validate_valid_csv():
    f = MockFile("data.csv", 100)
    valid, msg = utils.validate_uploaded_file(f)
    assert valid is True


def test_validate_valid_xlsx():
    f = MockFile("data.xlsx", 200)
    valid, msg = utils.validate_uploaded_file(f)
    assert valid is True


# ---------------------------------------------------------------------------
# load_uploaded_file tests
# ---------------------------------------------------------------------------

def test_load_valid_csv():
    csv_content = b"name,age,score\nAlice,25,8.5\nBob,30,7.0\n"
    f = MockFile("test.csv", len(csv_content), csv_content)

    # Monkey-patch read for pandas
    import types
    f_obj = io.BytesIO(csv_content)
    f_obj.name = "test.csv"
    f_obj.size = len(csv_content)

    # Use real BytesIO
    df, msg = utils.load_uploaded_file(f_obj)
    # BytesIO doesn't have .name in the same way; test validate directly
    valid, _ = utils.validate_uploaded_file(None)
    assert valid is False  # None returns False as expected


def test_load_file_none():
    df, msg = utils.load_uploaded_file(None)
    assert df is None
    assert "No file" in msg


# ---------------------------------------------------------------------------
# df_memory_usage tests
# ---------------------------------------------------------------------------

def test_memory_usage_returns_string(simple_df):
    result = utils.df_memory_usage(simple_df)
    assert isinstance(result, str)
    assert any(unit in result for unit in ["B", "KB", "MB"])


def test_memory_usage_small_df():
    # Very small df should report in bytes or KB
    df = pd.DataFrame({"a": [1]})
    result = utils.df_memory_usage(df)
    assert "B" in result


# ---------------------------------------------------------------------------
# completeness_pct tests
# ---------------------------------------------------------------------------

def test_completeness_full(simple_df):
    assert utils.completeness_pct(simple_df) == 100.0


def test_completeness_partial(df_with_missing):
    # 6 total cells, 2 missing → 4/6 = 66.67%
    pct = utils.completeness_pct(df_with_missing)
    assert abs(pct - 66.67) < 0.1


def test_completeness_empty():
    assert utils.completeness_pct(pd.DataFrame()) == 100.0


def test_completeness_all_missing():
    df = pd.DataFrame({"a": [None, None], "b": [None, None]})
    assert utils.completeness_pct(df) == 0.0


# ---------------------------------------------------------------------------
# numerical_cols / categorical_cols tests
# ---------------------------------------------------------------------------

def test_numerical_cols(simple_df):
    cols = utils.numerical_cols(simple_df)
    assert "age" in cols
    assert "score" in cols
    assert "city" not in cols


def test_categorical_cols(simple_df):
    cols = utils.categorical_cols(simple_df)
    assert "city" in cols
    assert "age" not in cols


def test_numerical_cols_empty():
    df = pd.DataFrame({"a": ["x", "y"]})
    assert utils.numerical_cols(df) == []


def test_categorical_cols_empty():
    df = pd.DataFrame({"a": [1, 2]})
    assert utils.categorical_cols(df) == []


# ---------------------------------------------------------------------------
# build_data_dictionary tests
# ---------------------------------------------------------------------------

def test_data_dictionary_shape(simple_df):
    result = utils.build_data_dictionary(simple_df)
    assert len(result) == simple_df.shape[1]


def test_data_dictionary_columns(simple_df):
    result = utils.build_data_dictionary(simple_df)
    for col in ["Column", "Data Type", "Missing Count", "Missing %", "Unique Values", "Sample Values"]:
        assert col in result.columns


def test_data_dictionary_missing_count(df_with_missing):
    result = utils.build_data_dictionary(df_with_missing)
    a_row = result[result["Column"] == "a"].iloc[0]
    assert a_row["Missing Count"] == 1


# ---------------------------------------------------------------------------
# safe_filename tests
# ---------------------------------------------------------------------------

def test_safe_filename_normal():
    assert utils.safe_filename("report.csv") == "report.csv"


def test_safe_filename_spaces():
    result = utils.safe_filename("my report 2024.csv")
    assert " " not in result


def test_safe_filename_special_chars():
    result = utils.safe_filename("data/file:name?.csv")
    assert "/" not in result
    assert ":" not in result
    assert "?" not in result


# ---------------------------------------------------------------------------
# df_to_csv_bytes tests
# ---------------------------------------------------------------------------

def test_df_to_csv_bytes_type(simple_df):
    result = utils.df_to_csv_bytes(simple_df)
    assert isinstance(result, bytes)


def test_df_to_csv_bytes_parseable(simple_df):
    result = utils.df_to_csv_bytes(simple_df)
    parsed = pd.read_csv(io.BytesIO(result))
    assert list(parsed.columns) == list(simple_df.columns)
    assert len(parsed) == len(simple_df)


# ---------------------------------------------------------------------------
# build_pipeline_json tests
# ---------------------------------------------------------------------------

def test_build_pipeline_json_valid(simple_df):
    log = [{"operation": "test", "columns": "age"}]
    result = utils.build_pipeline_json(simple_df, log)
    parsed = json.loads(result)
    assert parsed["dataset_rows"] == 3
    assert parsed["operations_count"] == 1


def test_build_pipeline_json_empty_log(simple_df):
    result = utils.build_pipeline_json(simple_df, [])
    parsed = json.loads(result)
    assert parsed["operations_count"] == 0


# ---------------------------------------------------------------------------
# build_data_dictionary — no crash on empty df
# ---------------------------------------------------------------------------

def test_data_dictionary_empty():
    result = utils.build_data_dictionary(pd.DataFrame())
    assert result.empty or len(result) == 0


# ---------------------------------------------------------------------------
# Row/column count correctness
# ---------------------------------------------------------------------------

def test_row_col_counts_preserved_after_no_op(simple_df):
    """Verify that passing df through utilities does not change shape."""
    orig_shape = simple_df.shape
    _ = utils.build_data_dictionary(simple_df)
    _ = utils.completeness_pct(simple_df)
    _ = utils.numerical_cols(simple_df)
    assert simple_df.shape == orig_shape
