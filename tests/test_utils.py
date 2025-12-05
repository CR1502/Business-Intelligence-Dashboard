"""
Unit Tests for Utils Module

Tests all utility functions and classes following best practices.
Uses pytest framework for comprehensive testing.

Author: Craig
Date: December 2024
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from utils import (
    FileValidator, DataFrameValidator, ColumnValidator,
    format_number, format_percentage, safe_divide,
    get_column_types, detect_date_columns, clean_currency_column,
    truncate_string, get_memory_usage,
    CSVExporter, ExcelExporter, Config
)


# ============================================================================
# FIXTURES
# Reusable test data following DRY principle
# ============================================================================

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'age': [25, 30, 35, 40],
        'name': ['Alice', 'Bob', 'Charlie', 'David'],
        'salary': [50000, 60000, 70000, 80000],
        'date': pd.date_range('2024-01-01', periods=4)
    })


@pytest.fixture
def empty_dataframe():
    """Create an empty DataFrame for testing."""
    return pd.DataFrame()


@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write('col1,col2\n1,2\n3,4\n')
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def temp_xlsx_file():
    """Create a temporary Excel file."""
    temp_path = tempfile.mktemp(suffix='.xlsx')
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    df.to_excel(temp_path, index=False)
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


# ============================================================================
# VALIDATOR TESTS
# ============================================================================

class TestFileValidator:
    """Test suite for FileValidator class."""

    def test_validate_existing_csv(self, temp_csv_file):
        """Test validation of existing CSV file."""
        validator = FileValidator()
        assert validator.validate(temp_csv_file) is True

    def test_validate_existing_xlsx(self, temp_xlsx_file):
        """Test validation of existing Excel file."""
        validator = FileValidator()
        assert validator.validate(temp_xlsx_file) is True

    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        validator = FileValidator()
        with pytest.raises(FileNotFoundError):
            validator.validate('nonexistent_file.csv')

    def test_validate_unsupported_format(self):
        """Test validation of unsupported file format."""
        validator = FileValidator()
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                validator.validate(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_supported_formats(self):
        """Test that all expected formats are supported."""
        validator = FileValidator()
        expected_formats = {'.csv', '.xlsx', '.xls', '.parquet', '.json', '.tsv'}
        assert validator.SUPPORTED_FORMATS == expected_formats


class TestDataFrameValidator:
    """Test suite for DataFrameValidator class."""

    def test_validate_valid_dataframe(self, sample_dataframe):
        """Test validation of valid DataFrame."""
        validator = DataFrameValidator()
        assert validator.validate(sample_dataframe) is True

    def test_validate_empty_dataframe(self, empty_dataframe):
        """Test validation of empty DataFrame."""
        validator = DataFrameValidator()
        with pytest.raises(ValueError, match="DataFrame is empty"):
            validator.validate(empty_dataframe)

    def test_validate_none_dataframe(self):
        """Test validation of None DataFrame."""
        validator = DataFrameValidator()
        with pytest.raises(ValueError, match="DataFrame cannot be None"):
            validator.validate(None)

    def test_validate_wrong_type(self):
        """Test validation of wrong data type."""
        validator = DataFrameValidator()
        with pytest.raises(ValueError, match="Expected pandas DataFrame"):
            validator.validate([1, 2, 3])


class TestColumnValidator:
    """Test suite for ColumnValidator class."""

    def test_validate_existing_column(self, sample_dataframe):
        """Test validation of existing column."""
        validator = ColumnValidator()
        assert validator.validate(sample_dataframe, 'age') is True

    def test_validate_existing_columns_list(self, sample_dataframe):
        """Test validation of multiple existing columns."""
        validator = ColumnValidator()
        assert validator.validate(sample_dataframe, ['age', 'name']) is True

    def test_validate_missing_column(self, sample_dataframe):
        """Test validation of missing column."""
        validator = ColumnValidator()
        with pytest.raises(ValueError, match="Columns not found"):
            validator.validate(sample_dataframe, 'nonexistent')

    def test_validate_partial_missing_columns(self, sample_dataframe):
        """Test validation with some missing columns."""
        validator = ColumnValidator()
        with pytest.raises(ValueError, match="Columns not found"):
            validator.validate(sample_dataframe, ['age', 'nonexistent'])


# ============================================================================
# FORMATTING FUNCTION TESTS
# ============================================================================

class TestFormatNumber:
    """Test suite for format_number function."""

    def test_format_integer(self):
        """Test formatting integer."""
        assert format_number(1234567) == "1,234,567"

    def test_format_float(self):
        """Test formatting float."""
        assert format_number(1234567.89) == "1,234,567.89"

    def test_format_with_decimals(self):
        """Test formatting with specific decimal places."""
        assert format_number(1234.5678, decimals=3) == "1,234.568"

    def test_format_nan(self):
        """Test formatting NaN value."""
        assert format_number(np.nan) == "N/A"

    def test_format_none(self):
        """Test formatting None value."""
        assert format_number(None) == "N/A"


class TestFormatPercentage:
    """Test suite for format_percentage function."""

    def test_format_valid_percentage(self):
        """Test formatting valid percentage."""
        assert format_percentage(0.456) == "45.60%"

    def test_format_zero_percentage(self):
        """Test formatting zero percentage."""
        assert format_percentage(0.0) == "0.00%"

    def test_format_one_hundred_percent(self):
        """Test formatting 100%."""
        assert format_percentage(1.0) == "100.00%"

    def test_format_nan_percentage(self):
        """Test formatting NaN percentage."""
        assert format_percentage(np.nan) == "N/A"

    def test_format_custom_decimals(self):
        """Test formatting with custom decimal places."""
        assert format_percentage(0.12345, decimals=3) == "12.345%"


class TestSafeDivide:
    """Test suite for safe_divide function."""

    def test_normal_division(self):
        """Test normal division."""
        assert safe_divide(10, 2) == 5.0

    def test_division_by_zero(self):
        """Test division by zero returns default."""
        assert safe_divide(10, 0, default=0.0) == 0.0

    def test_division_by_nan(self):
        """Test division by NaN returns default."""
        assert safe_divide(10, np.nan, default=-1.0) == -1.0

    def test_custom_default(self):
        """Test custom default value."""
        assert safe_divide(10, 0, default=999) == 999


# ============================================================================
# DATA ANALYSIS FUNCTION TESTS
# ============================================================================

class TestGetColumnTypes:
    """Test suite for get_column_types function."""

    def test_mixed_types(self, sample_dataframe):
        """Test getting column types from mixed DataFrame."""
        types = get_column_types(sample_dataframe)
        assert 'age' in types['numerical']
        assert 'salary' in types['numerical']
        assert 'name' in types['categorical']
        assert 'date' in types['datetime']

    def test_only_numerical(self):
        """Test DataFrame with only numerical columns."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3.0, 4.0]})
        types = get_column_types(df)
        assert len(types['numerical']) == 2
        assert len(types['categorical']) == 0

    def test_only_categorical(self):
        """Test DataFrame with only categorical columns."""
        df = pd.DataFrame({'a': ['x', 'y'], 'b': ['z', 'w']})
        types = get_column_types(df)
        assert len(types['categorical']) == 2
        assert len(types['numerical']) == 0


class TestDetectDateColumns:
    """Test suite for detect_date_columns function."""

    def test_detect_date_string_column(self):
        """Test detecting date strings."""
        df = pd.DataFrame({
            'date_col': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'text_col': ['abc', 'def', 'ghi']
        })
        date_cols = detect_date_columns(df)
        assert 'date_col' in date_cols
        assert 'text_col' not in date_cols

    def test_no_date_columns(self):
        """Test DataFrame without date columns."""
        df = pd.DataFrame({
            'num': [1, 2, 3],
            'text': ['a', 'b', 'c']
        })
        date_cols = detect_date_columns(df)
        assert len(date_cols) == 0


class TestCleanCurrencyColumn:
    """Test suite for clean_currency_column function."""

    def test_clean_dollar_signs(self):
        """Test cleaning dollar signs."""
        s = pd.Series(['$1,234.56', '$789.00', '$1,000.00'])
        result = clean_currency_column(s)
        expected = pd.Series([1234.56, 789.00, 1000.00])
        pd.testing.assert_series_equal(result, expected)

    def test_clean_spaces(self):
        """Test cleaning spaces in currency."""
        s = pd.Series(['$966 ', '$193 '])
        result = clean_currency_column(s)
        assert result[0] == 966.0
        assert result[1] == 193.0

    def test_handle_invalid_values(self):
        """Test handling invalid currency values."""
        s = pd.Series(['$100', 'invalid', '$200'])
        result = clean_currency_column(s)
        assert result[0] == 100.0
        assert pd.isna(result[1])
        assert result[2] == 200.0


class TestTruncateString:
    """Test suite for truncate_string function."""

    def test_truncate_long_string(self):
        """Test truncating long string."""
        text = "This is a very long text that needs truncation"
        result = truncate_string(text, max_length=20)
        assert len(result) == 20
        assert result.endswith("...")

    def test_no_truncation_needed(self):
        """Test string that doesn't need truncation."""
        text = "Short text"
        result = truncate_string(text, max_length=20)
        assert result == text

    def test_custom_suffix(self):
        """Test custom truncation suffix."""
        text = "Long text here"
        result = truncate_string(text, max_length=10, suffix=">>")
        assert result.endswith(">>")


class TestGetMemoryUsage:
    """Test suite for get_memory_usage function."""

    def test_small_dataframe(self):
        """Test memory usage of small DataFrame."""
        df = pd.DataFrame({'a': [1, 2, 3]})
        usage = get_memory_usage(df)
        assert 'B' in usage or 'KB' in usage

    def test_returns_string(self, sample_dataframe):
        """Test that function returns string."""
        usage = get_memory_usage(sample_dataframe)
        assert isinstance(usage, str)


# ============================================================================
# EXPORTER TESTS
# ============================================================================

class TestCSVExporter:
    """Test suite for CSVExporter class."""

    def test_export_csv(self, sample_dataframe):
        """Test exporting DataFrame to CSV."""
        exporter = CSVExporter()
        temp_path = tempfile.mktemp(suffix='.csv')

        try:
            result = exporter.export(sample_dataframe, temp_path)
            assert result is True
            assert os.path.exists(temp_path)

            # Verify content
            df_loaded = pd.read_csv(temp_path)
            assert df_loaded.shape == sample_dataframe.shape
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestExcelExporter:
    """Test suite for ExcelExporter class."""

    def test_export_excel(self, sample_dataframe):
        """Test exporting DataFrame to Excel."""
        exporter = ExcelExporter()
        temp_path = tempfile.mktemp(suffix='.xlsx')

        try:
            # Remove datetime column for Excel compatibility
            df_test = sample_dataframe.drop('date', axis=1)
            result = exporter.export(df_test, temp_path)
            assert result is True
            assert os.path.exists(temp_path)

            # Verify content
            df_loaded = pd.read_excel(temp_path)
            assert df_loaded.shape == df_test.shape
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


# ============================================================================
# CONFIG TESTS
# ============================================================================

class TestConfig:
    """Test suite for Config class."""

    def test_supported_formats_exists(self):
        """Test that supported formats are defined."""
        assert hasattr(Config, 'SUPPORTED_FILE_FORMATS')
        assert len(Config.SUPPORTED_FILE_FORMATS) > 0

    def test_display_settings_exist(self):
        """Test that display settings are defined."""
        assert hasattr(Config, 'MAX_DISPLAY_ROWS')
        assert hasattr(Config, 'MAX_STRING_LENGTH')
        assert hasattr(Config, 'DEFAULT_DECIMAL_PLACES')

    def test_config_values_valid(self):
        """Test that config values are valid."""
        assert Config.MAX_DISPLAY_ROWS > 0
        assert Config.MAX_STRING_LENGTH > 0
        assert Config.DEFAULT_DECIMAL_PLACES >= 0


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])