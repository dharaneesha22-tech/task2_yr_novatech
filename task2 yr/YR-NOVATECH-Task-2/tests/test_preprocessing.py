"""
tests/test_preprocessing.py

Unit tests for preprocessing.py using small synthetic DataFrames only.
These tests do NOT use the actual internship dataset.
"""

import pytest
import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import preprocessing as pp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def basic_df():
    """A simple DataFrame with numerical and categorical columns."""
    return pd.DataFrame({
        "age": [25, 30, np.nan, 40, 25],
        "salary": [50000, 60000, 70000, np.nan, 50000],
        "city": ["New York", "London", None, "Paris", "New York"],
        "score": [8.5, 7.0, 9.0, 6.5, 8.5],
    })


@pytest.fixture
def no_missing_df():
    return pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [10.0, 20.0, 30.0, 40.0, 50.0],
        "label": ["a", "b", "c", "a", "b"],
    })


@pytest.fixture
def duplicates_df():
    return pd.DataFrame({
        "a": [1, 2, 2, 3, 1],
        "b": ["x", "y", "y", "z", "x"],
    })


@pytest.fixture
def outlier_df():
    return pd.DataFrame({
        "value": [10, 12, 11, 13, 10, 11, 200, 9, 12, 11],
    })


# ---------------------------------------------------------------------------
# A. Missing value tests
# ---------------------------------------------------------------------------

def test_identify_missing_returns_correct_counts(basic_df):
    result = pp.identify_missing(basic_df)
    assert not result.empty
    missing_cols = result["column"].tolist()
    assert "age" in missing_cols
    assert "salary" in missing_cols
    assert "city" in missing_cols


def test_identify_missing_empty_df():
    result = pp.identify_missing(pd.DataFrame())
    assert result.empty


def test_impute_numerical_mean(basic_df):
    df_imp, stats = pp.impute_numerical(basic_df, ["age"], strategy="mean")
    assert df_imp["age"].isnull().sum() == 0
    # Mean of [25, 30, 40, 25] = 30.0
    expected_mean = np.mean([25, 30, 40, 25])
    assert abs(df_imp["age"].iloc[2] - expected_mean) < 1e-9


def test_impute_numerical_median(basic_df):
    df_imp, stats = pp.impute_numerical(basic_df, ["age"], strategy="median")
    assert df_imp["age"].isnull().sum() == 0


def test_impute_numerical_constant(basic_df):
    df_imp, stats = pp.impute_numerical(basic_df, ["salary"], strategy="constant", fill_value=-1.0)
    assert df_imp["salary"].isnull().sum() == 0
    assert df_imp["salary"].iloc[3] == -1.0


def test_impute_numerical_invalid_strategy(basic_df):
    with pytest.raises(ValueError, match="Unknown strategy"):
        pp.impute_numerical(basic_df, ["age"], strategy="random")


def test_impute_numerical_constant_no_value(basic_df):
    with pytest.raises(ValueError, match="fill_value must be provided"):
        pp.impute_numerical(basic_df, ["age"], strategy="constant", fill_value=None)


def test_impute_categorical_mode(basic_df):
    df_imp, stats = pp.impute_categorical(basic_df, ["city"], strategy="mode")
    assert df_imp["city"].isnull().sum() == 0


def test_impute_categorical_constant(basic_df):
    df_imp, stats = pp.impute_categorical(basic_df, ["city"], strategy="constant", fill_value="Unknown")
    assert df_imp["city"].isnull().sum() == 0
    assert df_imp["city"].iloc[2] == "Unknown"


def test_impute_categorical_invalid_strategy(basic_df):
    with pytest.raises(ValueError, match="Unknown strategy"):
        pp.impute_categorical(basic_df, ["city"], strategy="median")


def test_drop_rows_with_missing(basic_df):
    df_clean, dropped = pp.drop_rows_with_missing(basic_df)
    assert dropped > 0
    assert df_clean.isnull().sum().sum() == 0


def test_drop_rows_with_missing_specific_columns(basic_df):
    df_clean, dropped = pp.drop_rows_with_missing(basic_df, columns=["age"])
    # Only 1 row has NaN in 'age'
    assert dropped == 1
    assert df_clean["age"].isnull().sum() == 0


def test_drop_columns_with_missing():
    df = pd.DataFrame({
        "a": [1, 2, None, None, None],  # 60% missing
        "b": [1, 2, 3, 4, 5],
    })
    new_df, dropped = pp.drop_columns_with_missing(df, threshold_pct=50.0)
    assert "a" in dropped
    assert "b" not in dropped


# ---------------------------------------------------------------------------
# B. Duplicate tests
# ---------------------------------------------------------------------------

def test_count_duplicates(duplicates_df):
    count = pp.count_duplicates(duplicates_df)
    # rows (1,"x") and (2,"y") appear twice each → 2 duplicates
    assert count == 2


def test_count_duplicates_no_dups(no_missing_df):
    assert pp.count_duplicates(no_missing_df) == 0


def test_remove_duplicates(duplicates_df):
    df_clean, removed = pp.remove_duplicates(duplicates_df)
    assert removed == 2
    assert pp.count_duplicates(df_clean) == 0
    assert len(df_clean) == 3


def test_remove_duplicates_empty():
    df, removed = pp.remove_duplicates(pd.DataFrame())
    assert removed == 0


# ---------------------------------------------------------------------------
# C. Data type tests
# ---------------------------------------------------------------------------

def test_convert_to_numeric_success():
    df = pd.DataFrame({"val": ["1", "2", "3"]})
    new_df, results = pp.convert_to_numeric(df, ["val"])
    assert pd.api.types.is_numeric_dtype(new_df["val"])
    assert results["val"]["coerced_nulls"] == 0


def test_convert_to_numeric_with_invalid():
    df = pd.DataFrame({"val": ["1", "abc", "3"]})
    new_df, results = pp.convert_to_numeric(df, ["val"])
    assert results["val"]["coerced_nulls"] == 1
    assert pd.isna(new_df["val"].iloc[1])


def test_convert_to_numeric_missing_column():
    df = pd.DataFrame({"a": [1, 2, 3]})
    new_df, results = pp.convert_to_numeric(df, ["nonexistent"])
    assert results["nonexistent"]["converted"] == False


def test_parse_datetime_column():
    df = pd.DataFrame({"date": ["2021-01-01", "2021-06-15", "not-a-date"]})
    new_df, errors, msg = pp.parse_datetime_column(df, "date")
    assert errors == 1
    assert pd.api.types.is_datetime64_any_dtype(new_df["date"])


def test_parse_datetime_missing_column():
    df = pd.DataFrame({"a": [1, 2]})
    new_df, errors, msg = pp.parse_datetime_column(df, "missing_col")
    assert "not found" in msg


# ---------------------------------------------------------------------------
# D. Text normalisation tests
# ---------------------------------------------------------------------------

def test_trim_whitespace():
    df = pd.DataFrame({"name": ["  Alice  ", " Bob", "Carol "]})
    new_df = pp.trim_whitespace(df, ["name"])
    assert new_df["name"].tolist() == ["Alice", "Bob", "Carol"]


def test_normalize_whitespace():
    df = pd.DataFrame({"text": ["hello   world", "foo  bar"]})
    new_df = pp.normalize_whitespace(df, ["text"])
    assert new_df["text"].tolist() == ["hello world", "foo bar"]


def test_standardize_casing_lower():
    df = pd.DataFrame({"city": ["New York", "LONDON", "Paris"]})
    new_df = pp.standardize_casing(df, ["city"], case="lower")
    assert new_df["city"].tolist() == ["new york", "london", "paris"]


def test_standardize_casing_upper():
    df = pd.DataFrame({"city": ["New York", "London"]})
    new_df = pp.standardize_casing(df, ["city"], case="upper")
    assert new_df["city"].tolist() == ["NEW YORK", "LONDON"]


# ---------------------------------------------------------------------------
# E. Outlier tests
# ---------------------------------------------------------------------------

def test_detect_outliers_iqr(outlier_df):
    result = pp.detect_outliers_iqr(outlier_df, ["value"])
    assert "value" in result
    assert result["value"]["outlier_count"] >= 1  # 200 is clearly an outlier


def test_detect_outliers_empty_df():
    result = pp.detect_outliers_iqr(pd.DataFrame({"a": []}), ["a"])
    assert result == {}


def test_handle_outliers_cap(outlier_df):
    new_df, stats = pp.handle_outliers(outlier_df, ["value"], method="cap")
    # After capping, max should be the upper bound, not 200
    assert new_df["value"].max() < 200


def test_handle_outliers_remove(outlier_df):
    before_len = len(outlier_df)
    new_df, stats = pp.handle_outliers(outlier_df, ["value"], method="remove")
    assert len(new_df) < before_len


# ---------------------------------------------------------------------------
# F. Column operations tests
# ---------------------------------------------------------------------------

def test_drop_columns(no_missing_df):
    new_df, dropped = pp.drop_columns(no_missing_df, ["x", "y"])
    assert "x" not in new_df.columns
    assert "y" not in new_df.columns
    assert dropped == ["x", "y"]


def test_drop_columns_nonexistent(no_missing_df):
    new_df, dropped = pp.drop_columns(no_missing_df, ["nonexistent"])
    assert dropped == []
    assert list(new_df.columns) == list(no_missing_df.columns)


def test_rename_column(no_missing_df):
    new_df, msg = pp.rename_column(no_missing_df, "x", "feature_x")
    assert "feature_x" in new_df.columns
    assert "x" not in new_df.columns


def test_rename_column_duplicate_name(no_missing_df):
    _, msg = pp.rename_column(no_missing_df, "x", "y")
    assert "already exists" in msg


def test_rename_column_not_found(no_missing_df):
    _, msg = pp.rename_column(no_missing_df, "missing", "new_name")
    assert "not found" in msg


def test_rename_column_empty_name(no_missing_df):
    _, msg = pp.rename_column(no_missing_df, "x", "")
    assert "cannot be empty" in msg


# ---------------------------------------------------------------------------
# G. Feature engineering tests
# ---------------------------------------------------------------------------

def test_create_arithmetic_feature_add(no_missing_df):
    new_df, msg = pp.create_arithmetic_feature(no_missing_df, "x", "y", "+", "x_plus_y")
    assert "x_plus_y" in new_df.columns
    assert new_df["x_plus_y"].iloc[0] == 1 + 10.0


def test_create_arithmetic_feature_divide(no_missing_df):
    new_df, msg = pp.create_arithmetic_feature(no_missing_df, "y", "x", "/", "y_div_x")
    assert "y_div_x" in new_df.columns
    assert new_df["y_div_x"].iloc[0] == 10.0


def test_create_arithmetic_feature_division_by_zero():
    df = pd.DataFrame({"a": [10, 20, 30], "b": [0, 0, 0]})
    new_df, msg = pp.create_arithmetic_feature(df, "a", "b", "/", "result")
    # Division by zero → NaN, not an exception
    assert new_df["result"].isnull().all()


def test_create_arithmetic_feature_duplicate_name(no_missing_df):
    _, msg = pp.create_arithmetic_feature(no_missing_df, "x", "y", "+", "x")
    assert "already exists" in msg


def test_create_arithmetic_feature_col_not_found(no_missing_df):
    _, msg = pp.create_arithmetic_feature(no_missing_df, "nonexistent", "y", "+", "new_col")
    assert "not found" in msg


def test_create_arithmetic_feature_invalid_op(no_missing_df):
    _, msg = pp.create_arithmetic_feature(no_missing_df, "x", "y", "^", "new_col")
    assert "Unsupported operation" in msg


def test_create_arithmetic_feature_empty_name(no_missing_df):
    _, msg = pp.create_arithmetic_feature(no_missing_df, "x", "y", "+", "")
    assert "cannot be empty" in msg


def test_extract_datetime_features():
    df = pd.DataFrame({"date": pd.to_datetime(["2021-03-15", "2022-07-04"])})
    new_df, msg = pp.extract_datetime_features(df, "date", ["year", "month", "day"])
    assert "date_year" in new_df.columns
    assert new_df["date_year"].iloc[0] == 2021
    assert new_df["date_month"].iloc[1] == 7


def test_extract_datetime_features_col_not_found():
    df = pd.DataFrame({"a": [1, 2]})
    _, msg = pp.extract_datetime_features(df, "missing", ["year"])
    assert "not found" in msg


def test_create_bins():
    df = pd.DataFrame({"age": [10, 25, 35, 55, 70]})
    new_df, msg = pp.create_bins(df, "age", bins=3, labels=None, new_col_name="age_group")
    assert "age_group" in new_df.columns
    assert new_df["age_group"].isnull().sum() == 0


def test_create_bins_duplicate_col_name():
    df = pd.DataFrame({"age": [10, 25, 35]})
    _, msg = pp.create_bins(df, "age", bins=3, labels=None, new_col_name="age")
    assert "already exists" in msg


def test_create_bins_empty_name():
    df = pd.DataFrame({"age": [10, 25, 35]})
    _, msg = pp.create_bins(df, "age", bins=3, labels=None, new_col_name="")
    assert "cannot be empty" in msg


def test_create_interaction_feature(no_missing_df):
    new_df, msg = pp.create_interaction_feature(no_missing_df, "x", "y", "x_times_y")
    assert "x_times_y" in new_df.columns
    assert new_df["x_times_y"].iloc[0] == 1 * 10.0


# ---------------------------------------------------------------------------
# Original dataset preservation
# ---------------------------------------------------------------------------

def test_original_df_not_modified_by_imputation(basic_df):
    original_copy = basic_df.copy()
    pp.impute_numerical(basic_df, ["age"], strategy="mean")
    pd.testing.assert_frame_equal(basic_df, original_copy)


def test_original_df_not_modified_by_outlier_handling(outlier_df):
    original_copy = outlier_df.copy()
    pp.handle_outliers(outlier_df, ["value"], method="cap")
    pd.testing.assert_frame_equal(outlier_df, original_copy)


def test_original_df_not_modified_by_drop(duplicates_df):
    original_copy = duplicates_df.copy()
    pp.remove_duplicates(duplicates_df)
    pd.testing.assert_frame_equal(duplicates_df, original_copy)
