"""
Unit Tests for Data Processor Module

Comprehensive tests for all data processing functionality including
Strategy Pattern implementation, data loading, cleaning, and filtering.

Author: Craig
Date: December 2024
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from datetime import datetime

from data_processor import (
    DataLoadStrategy, CSVLoadStrategy, ExcelLoadStrategy,
    JSONLoadStrategy, ParquetLoadStrategy,
    DataLoader, DataCleaner, DataProfiler, DataFilter, DataProcessor
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 40, 45],
        'salary': [50000, 60000, 70000, 80000, 90000],
        'department': ['HR', 'IT', 'IT', 'Finance', 'HR'],
        'hire_date': pd.date_range('2020-01-01', periods=5)
    })


@pytest.fixture
def dataframe_with_missing():
    """Create DataFrame with missing values."""
    return pd.DataFrame({
        'col1': [1, 2, np.nan, 4, 5],
        'col2': ['a', 'b', 'c', np.nan, 'e'],
        'col3': [10.5, np.nan, 30.5, 40.5, 50.5]
    })


@pytest.fixture
def dataframe_with_duplicates():
    """Create DataFrame with duplicate rows."""
    return pd.DataFrame({
        'id': [1, 2, 3, 2, 4],
        'value': ['a', 'b', 'c', 'b', 'd']
    })


@pytest.fixture
def temp_csv_file(sample_dataframe):
    """Create temporary CSV file."""
    temp_path = tempfile.mktemp(suffix='.csv')
    sample_dataframe.to_csv(temp_path, index=False)
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def temp_excel_file(sample_dataframe):
    """Create temporary Excel file."""
    temp_path = tempfile.mktemp(suffix='.xlsx')
    sample_dataframe.to_excel(temp_path, index=False)
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def temp_json_file(sample_dataframe):
    """Create temporary JSON file."""
    temp_path = tempfile.mktemp(suffix='.json')
    # Drop datetime column for JSON compatibility
    df_json = sample_dataframe.drop('hire_date', axis=1)
    df_json.to_json(temp_path)
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


# ============================================================================
# STRATEGY PATTERN TESTS
# ============================================================================

class TestCSVLoadStrategy:
    """Test suite for CSVLoadStrategy."""

    def test_can_handle_csv(self):
        """Test CSV file detection."""
        strategy = CSVLoadStrategy()
        assert strategy.can_handle('file.csv') is True
        assert strategy.can_handle('file.CSV') is True
        assert strategy.can_handle('file.xlsx') is False

    def test_load_csv(self, temp_csv_file):
        """Test loading CSV file."""
        strategy = CSVLoadStrategy()
        df = strategy.load(temp_csv_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_nonexistent_csv(self):
        """Test loading non-existent CSV file."""
        strategy = CSVLoadStrategy()
        with pytest.raises(Exception):
            strategy.load('nonexistent.csv')


class TestExcelLoadStrategy:
    """Test suite for ExcelLoadStrategy."""

    def test_can_handle_excel(self):
        """Test Excel file detection."""
        strategy = ExcelLoadStrategy()
        assert strategy.can_handle('file.xlsx') is True
        assert strategy.can_handle('file.xls') is True
        assert strategy.can_handle('file.XLSX') is True
        assert strategy.can_handle('file.csv') is False

    def test_load_excel(self, temp_excel_file):
        """Test loading Excel file."""
        strategy = ExcelLoadStrategy()
        df = strategy.load(temp_excel_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


class TestJSONLoadStrategy:
    """Test suite for JSONLoadStrategy."""

    def test_can_handle_json(self):
        """Test JSON file detection."""
        strategy = JSONLoadStrategy()
        assert strategy.can_handle('file.json') is True
        assert strategy.can_handle('file.JSON') is True
        assert strategy.can_handle('file.csv') is False

    def test_load_json(self, temp_json_file):
        """Test loading JSON file."""
        strategy = JSONLoadStrategy()
        df = strategy.load(temp_json_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


class TestParquetLoadStrategy:
    """Test suite for ParquetLoadStrategy."""

    def test_can_handle_parquet(self):
        """Test Parquet file detection."""
        strategy = ParquetLoadStrategy()
        assert strategy.can_handle('file.parquet') is True
        assert strategy.can_handle('file.PARQUET') is True
        assert strategy.can_handle('file.csv') is False


# ============================================================================
# DATA LOADER TESTS
# ============================================================================

class TestDataLoader:
    """Test suite for DataLoader class."""

    def test_initialization(self):
        """Test DataLoader initialization."""
        loader = DataLoader()
        assert len(loader.strategies) >= 4

    def test_load_csv(self, temp_csv_file):
        """Test loading CSV through DataLoader."""
        loader = DataLoader()
        df = loader.load_data(temp_csv_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_load_excel(self, temp_excel_file):
        """Test loading Excel through DataLoader."""
        loader = DataLoader()
        df = loader.load_data(temp_excel_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_load_json(self, temp_json_file):
        """Test loading JSON through DataLoader."""
        loader = DataLoader()
        df = loader.load_data(temp_json_file)
        assert isinstance(df, pd.DataFrame)

    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        loader = DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_data('nonexistent.csv')

    def test_add_strategy(self):
        """Test adding new strategy."""
        loader = DataLoader()
        initial_count = len(loader.strategies)

        # Create mock strategy
        class MockStrategy(DataLoadStrategy):
            def can_handle(self, filepath):
                return False

            def load(self, filepath):
                return pd.DataFrame()

        loader.add_strategy(MockStrategy())
        assert len(loader.strategies) == initial_count + 1


# ============================================================================
# DATA CLEANER TESTS
# ============================================================================

class TestDataCleaner:
    """Test suite for DataCleaner class."""

    def test_handle_missing_none(self, dataframe_with_missing):
        """Test 'none' strategy - no changes."""
        df_cleaned = DataCleaner.handle_missing_values(dataframe_with_missing, strategy='none')
        assert df_cleaned.isnull().sum().sum() == dataframe_with_missing.isnull().sum().sum()

    def test_handle_missing_drop(self, dataframe_with_missing):
        """Test dropping rows with missing values."""
        df_cleaned = DataCleaner.handle_missing_values(dataframe_with_missing, strategy='drop')
        assert df_cleaned.isnull().sum().sum() == 0
        assert len(df_cleaned) < len(dataframe_with_missing)

    def test_handle_missing_fill_mean(self, dataframe_with_missing):
        """Test filling with mean."""
        df_cleaned = DataCleaner.handle_missing_values(dataframe_with_missing, strategy='fill_mean')
        numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            assert df_cleaned[col].isnull().sum() == 0

    def test_handle_missing_fill_median(self, dataframe_with_missing):
        """Test filling with median."""
        df_cleaned = DataCleaner.handle_missing_values(dataframe_with_missing, strategy='fill_median')
        numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            assert df_cleaned[col].isnull().sum() == 0

    def test_handle_missing_fill_mode(self, dataframe_with_missing):
        """Test filling with mode."""
        df_cleaned = DataCleaner.handle_missing_values(dataframe_with_missing, strategy='fill_mode')
        # Check categorical columns are filled
        assert df_cleaned['col2'].isnull().sum() == 0

    def test_convert_data_types(self):
        """Test automatic data type conversion."""
        df = pd.DataFrame({
            'price': ['$100', '$200', '$300'],
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'bool_col': ['TRUE', 'FALSE', 'TRUE']
        })

        df_converted = DataCleaner.convert_data_types(df)

        # Check currency conversion
        assert pd.api.types.is_numeric_dtype(df_converted['price'])

        # Check date conversion
        assert pd.api.types.is_datetime64_any_dtype(df_converted['date'])

    def test_remove_duplicates(self, dataframe_with_duplicates):
        """Test removing duplicate rows."""
        df_cleaned = DataCleaner.remove_duplicates(dataframe_with_duplicates)
        assert len(df_cleaned) < len(dataframe_with_duplicates)
        assert df_cleaned.duplicated().sum() == 0

    def test_handle_outliers_zscore(self):
        """Test outlier removal using z-score."""
        df = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 100]  # 100 is an outlier
        })

        df_cleaned = DataCleaner.handle_outliers(df, ['values'], method='zscore', threshold=2.0)
        assert len(df_cleaned) < len(df)
        assert 100 not in df_cleaned['values'].values

    def test_handle_outliers_iqr(self):
        """Test outlier removal using IQR."""
        df = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 100]  # 100 is an outlier
        })

        df_cleaned = DataCleaner.handle_outliers(df, ['values'], method='iqr', threshold=1.5)
        assert len(df_cleaned) < len(df)


# ============================================================================
# DATA PROFILER TESTS
# ============================================================================

class TestDataProfiler:
    """Test suite for DataProfiler class."""

    def test_initialization(self, sample_dataframe):
        """Test DataProfiler initialization."""
        profiler = DataProfiler(sample_dataframe)
        assert profiler.df is not None

    def test_initialization_empty_dataframe(self):
        """Test initialization with empty DataFrame."""
        with pytest.raises(ValueError):
            DataProfiler(pd.DataFrame())

    def test_get_basic_info(self, sample_dataframe):
        """Test getting basic info."""
        profiler = DataProfiler(sample_dataframe)
        info = profiler.get_basic_info()

        assert info['rows'] == 5
        assert info['columns'] == 6
        assert 'column_names' in info
        assert 'data_types' in info
        assert 'memory_usage' in info

    def test_get_missing_values_report(self, dataframe_with_missing):
        """Test missing values report."""
        profiler = DataProfiler(dataframe_with_missing)
        report = profiler.get_missing_values_report()

        assert isinstance(report, pd.DataFrame)
        assert len(report) > 0
        assert 'Missing_Count' in report.columns
        assert 'Missing_Percentage' in report.columns

    def test_get_numerical_summary(self, sample_dataframe):
        """Test numerical summary statistics."""
        profiler = DataProfiler(sample_dataframe)
        summary = profiler.get_numerical_summary()

        assert isinstance(summary, pd.DataFrame)
        assert 'age' in summary.columns
        assert 'salary' in summary.columns

    def test_get_categorical_summary(self, sample_dataframe):
        """Test categorical summary statistics."""
        profiler = DataProfiler(sample_dataframe)
        summary = profiler.get_categorical_summary()

        assert isinstance(summary, dict)
        assert 'department' in summary
        assert 'unique_count' in summary['department']
        assert 'top_value' in summary['department']

    def test_get_correlation_matrix(self, sample_dataframe):
        """Test correlation matrix generation."""
        profiler = DataProfiler(sample_dataframe)
        corr_matrix = profiler.get_correlation_matrix()

        assert isinstance(corr_matrix, pd.DataFrame)
        assert 'age' in corr_matrix.columns
        assert 'salary' in corr_matrix.columns

    def test_get_full_profile(self, sample_dataframe):
        """Test full profile generation."""
        profiler = DataProfiler(sample_dataframe)
        profile = profiler.get_full_profile()

        assert 'basic_info' in profile
        assert 'missing_values' in profile
        assert 'numerical_summary' in profile
        assert 'categorical_summary' in profile
        assert 'correlation_matrix' in profile


# ============================================================================
# DATA FILTER TESTS
# ============================================================================

class TestDataFilter:
    """Test suite for DataFilter class."""

    def test_filter_numerical_min(self, sample_dataframe):
        """Test filtering with minimum value."""
        filtered = DataFilter.filter_numerical(sample_dataframe, 'age', min_val=30)
        assert len(filtered) == 4
        assert filtered['age'].min() >= 30

    def test_filter_numerical_max(self, sample_dataframe):
        """Test filtering with maximum value."""
        filtered = DataFilter.filter_numerical(sample_dataframe, 'age', max_val=35)
        assert len(filtered) == 3
        assert filtered['age'].max() <= 35

    def test_filter_numerical_range(self, sample_dataframe):
        """Test filtering with range."""
        filtered = DataFilter.filter_numerical(sample_dataframe, 'age', min_val=30, max_val=40)
        assert len(filtered) == 3
        assert filtered['age'].min() >= 30
        assert filtered['age'].max() <= 40

    def test_filter_categorical(self, sample_dataframe):
        """Test categorical filtering."""
        filtered = DataFilter.filter_categorical(sample_dataframe, 'department', ['IT', 'HR'])
        assert len(filtered) == 4
        assert all(filtered['department'].isin(['IT', 'HR']))

    def test_filter_categorical_empty_values(self, sample_dataframe):
        """Test categorical filtering with empty values list."""
        filtered = DataFilter.filter_categorical(sample_dataframe, 'department', [])
        assert len(filtered) == len(sample_dataframe)

    def test_filter_date_range(self, sample_dataframe):
        """Test date range filtering."""
        start_date = pd.Timestamp('2020-01-02')
        end_date = pd.Timestamp('2020-01-04')

        filtered = DataFilter.filter_date_range(sample_dataframe, 'hire_date', start_date, end_date)
        assert len(filtered) == 3

    def test_apply_multiple_filters(self, sample_dataframe):
        """Test applying multiple filters."""
        filters = [
            {'type': 'numerical', 'column': 'age', 'min_val': 30, 'max_val': 40},
            {'type': 'categorical', 'column': 'department', 'values': ['IT', 'Finance']}
        ]

        filtered = DataFilter.apply_multiple_filters(sample_dataframe, filters)
        assert len(filtered) <= len(sample_dataframe)

    def test_filter_invalid_column(self, sample_dataframe):
        """Test filtering with invalid column."""
        with pytest.raises(ValueError):
            DataFilter.filter_numerical(sample_dataframe, 'nonexistent', min_val=0)


# ============================================================================
# DATA PROCESSOR TESTS (Facade)
# ============================================================================

class TestDataProcessor:
    """Test suite for DataProcessor class (Facade)."""

    def test_initialization(self):
        """Test DataProcessor initialization."""
        processor = DataProcessor()