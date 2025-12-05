"""
Utility Module for Business Intelligence Dashboard

This module provides helper functions and utilities following SOLID principles.
Implements Single Responsibility Principle - each function has one clear purpose.

Author: Craig
Date: December 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, List, Any
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# INTERFACE SEGREGATION PRINCIPLE (ISP)
# Define specific interfaces for different validation types
# ============================================================================

class DataValidator(ABC):
    """
    Abstract base class for data validation.
    Follows Interface Segregation Principle - clients depend only on methods they use.
    """

    @abstractmethod
    def validate(self, data: Any) -> bool:
        """
        Validate the given data.

        Args:
            data: Data to validate

        Returns:
            bool: True if validation passes, False otherwise
        """
        pass


class FileValidator(DataValidator):
    """
    Validates file existence and format.
    Follows Single Responsibility Principle - only handles file validation.
    """

    SUPPORTED_FORMATS = {'.csv', '.xlsx', '.xls', '.parquet', '.json', '.tsv'}

    def validate(self, file_path: Union[str, Path]) -> bool:
        """
        Validate if file exists and has supported format.

        Args:
            file_path: Path to the file

        Returns:
            bool: True if file is valid, False otherwise

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        path = Path(file_path)

        if not path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            logger.error(f"Unsupported format: {path.suffix}")
            raise ValueError(
                f"Unsupported file format: {path.suffix}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        logger.info(f"File validation passed: {file_path}")
        return True


class DataFrameValidator(DataValidator):
    """
    Validates pandas DataFrame properties.
    Follows Single Responsibility Principle - only handles DataFrame validation.
    """

    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validate if DataFrame is valid and not empty.

        Args:
            df: DataFrame to validate

        Returns:
            bool: True if DataFrame is valid, False otherwise

        Raises:
            ValueError: If DataFrame is None or empty
        """
        if df is None:
            logger.error("DataFrame is None")
            raise ValueError("DataFrame cannot be None")

        if not isinstance(df, pd.DataFrame):
            logger.error(f"Expected DataFrame, got {type(df)}")
            raise ValueError(f"Expected pandas DataFrame, got {type(df)}")

        if df.empty:
            logger.error("DataFrame is empty")
            raise ValueError("DataFrame is empty")

        logger.info(f"DataFrame validation passed: {df.shape[0]} rows, {df.shape[1]} columns")
        return True


class ColumnValidator(DataValidator):
    """
    Validates column existence in DataFrame.
    Follows Single Responsibility Principle - only handles column validation.
    """

    def validate(self, df: pd.DataFrame, columns: Union[str, List[str]]) -> bool:
        """
        Validate if specified columns exist in DataFrame.

        Args:
            df: DataFrame to check
            columns: Column name(s) to validate

        Returns:
            bool: True if all columns exist, False otherwise

        Raises:
            ValueError: If any column doesn't exist
        """
        if isinstance(columns, str):
            columns = [columns]

        missing_columns = [col for col in columns if col not in df.columns]

        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            raise ValueError(
                f"Columns not found in DataFrame: {', '.join(missing_columns)}"
            )

        logger.info(f"Column validation passed: {columns}")
        return True


# ============================================================================
# UTILITY FUNCTIONS
# These follow Single Responsibility Principle
# ============================================================================

def format_number(number: Union[int, float], decimals: int = 2) -> str:
    """
    Format a number for display with thousand separators.

    Args:
        number: Number to format
        decimals: Number of decimal places

    Returns:
        str: Formatted number string

    Example:
        >>> format_number(1234567.89)
        '1,234,567.89'
    """
    try:
        if pd.isna(number):
            return "N/A"

        if isinstance(number, (int, np.integer)):
            return f"{number:,}"

        return f"{number:,.{decimals}f}"
    except (ValueError, TypeError) as e:
        logger.warning(f"Error formatting number {number}: {e}")
        return str(number)


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a value as percentage.

    Args:
        value: Value to format (0.5 = 50%)
        decimals: Number of decimal places

    Returns:
        str: Formatted percentage string

    Example:
        >>> format_percentage(0.456)
        '45.60%'
    """
    try:
        if pd.isna(value):
            return "N/A"
        return f"{value * 100:.{decimals}f}%"
    except (ValueError, TypeError) as e:
        logger.warning(f"Error formatting percentage {value}: {e}")
        return str(value)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if division by zero.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value to return if division fails

    Returns:
        float: Result of division or default value

    Example:
        >>> safe_divide(10, 2)
        5.0
        >>> safe_divide(10, 0, default=0)
        0.0
    """
    try:
        if denominator == 0 or pd.isna(denominator):
            return default
        return numerator / denominator
    except (ValueError, TypeError, ZeroDivisionError):
        return default


def get_column_types(df: pd.DataFrame) -> dict:
    """
    Categorize DataFrame columns by data type.

    Args:
        df: DataFrame to analyze

    Returns:
        dict: Dictionary with keys 'numerical', 'categorical', 'datetime'

    Example:
        >>> df = pd.DataFrame({'age': [25, 30], 'name': ['Alice', 'Bob']})
        >>> types = get_column_types(df)
        >>> types['numerical']
        ['age']
    """
    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime = df.select_dtypes(include=['datetime64']).columns.tolist()

    return {
        'numerical': numerical,
        'categorical': categorical,
        'datetime': datetime
    }


def detect_date_columns(df: pd.DataFrame, sample_size: int = 100) -> List[str]:
    """
    Detect columns that might contain date strings.

    Args:
        df: DataFrame to analyze
        sample_size: Number of rows to sample for detection

    Returns:
        List[str]: List of potential date column names
    """
    potential_date_cols = []

    for col in df.select_dtypes(include=['object']).columns:
        sample = df[col].dropna().head(sample_size)

        if len(sample) == 0:
            continue

        # Try to parse as dates
        try:
            pd.to_datetime(sample, errors='coerce')
            # If more than 50% parse successfully, consider it a date column
            parsed = pd.to_datetime(sample, errors='coerce')
            if parsed.notna().sum() / len(sample) > 0.5:
                potential_date_cols.append(col)
        except Exception:
            continue

    return potential_date_cols


def clean_currency_column(series: pd.Series) -> pd.Series:
    """
    Clean currency columns by removing symbols and converting to float.

    Args:
        series: Pandas Series with currency values

    Returns:
        pd.Series: Cleaned numeric series

    Example:
        >>> s = pd.Series(['$1,234.56', '$789.00'])
        >>> clean_currency_column(s)
        0    1234.56
        1     789.00
        dtype: float64
    """
    try:
        # Remove currency symbols, commas, and spaces
        cleaned = series.astype(str).str.replace(r'[$,€£¥\s]', '', regex=True)
        return pd.to_numeric(cleaned, errors='coerce')
    except Exception as e:
        logger.warning(f"Error cleaning currency column: {e}")
        return series


def truncate_string(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """
    Truncate a string to maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        str: Truncated string

    Example:
        >>> truncate_string("This is a very long text", 10)
        'This is...'
    """
    if not isinstance(text, str):
        text = str(text)

    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix


def get_memory_usage(df: pd.DataFrame) -> str:
    """
    Get human-readable memory usage of DataFrame.

    Args:
        df: DataFrame to analyze

    Returns:
        str: Memory usage string (e.g., "2.5 MB")
    """
    memory_bytes = df.memory_usage(deep=True).sum()

    for unit in ['B', 'KB', 'MB', 'GB']:
        if memory_bytes < 1024.0:
            return f"{memory_bytes:.2f} {unit}"
        memory_bytes /= 1024.0

    return f"{memory_bytes:.2f} TB"


# ============================================================================
# EXPORT UTILITIES
# These follow Single Responsibility Principle
# ============================================================================

class DataExporter(ABC):
    """
    Abstract base class for data export.
    Follows Open/Closed Principle - open for extension, closed for modification.
    """

    @abstractmethod
    def export(self, data: Any, filepath: Union[str, Path]) -> bool:
        """
        Export data to file.

        Args:
            data: Data to export
            filepath: Destination file path

        Returns:
            bool: True if export successful, False otherwise
        """
        pass


class CSVExporter(DataExporter):
    """
    Export DataFrame to CSV format.
    Follows Single Responsibility Principle.
    """

    def export(self, df: pd.DataFrame, filepath: Union[str, Path]) -> bool:
        """
        Export DataFrame to CSV file.

        Args:
            df: DataFrame to export
            filepath: Destination CSV file path

        Returns:
            bool: True if export successful, False otherwise
        """
        try:
            df.to_csv(filepath, index=False)
            logger.info(f"Successfully exported to CSV: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False


class ExcelExporter(DataExporter):
    """
    Export DataFrame to Excel format.
    Follows Single Responsibility Principle.
    """

    def export(self, df: pd.DataFrame, filepath: Union[str, Path]) -> bool:
        """
        Export DataFrame to Excel file.

        Args:
            df: DataFrame to export
            filepath: Destination Excel file path

        Returns:
            bool: True if export successful, False otherwise
        """
        try:
            df.to_excel(filepath, index=False, engine='openpyxl')
            logger.info(f"Successfully exported to Excel: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
            return False


# ============================================================================
# CONSTANTS
# Centralized configuration following DRY principle
# ============================================================================

class Config:
    """
    Configuration constants for the application.
    Centralized configuration following Single Responsibility Principle.
    """

    # File formats
    SUPPORTED_FILE_FORMATS = {'.csv', '.xlsx', '.xls', '.parquet', '.json', '.tsv'}

    # Display settings
    MAX_DISPLAY_ROWS = 100
    MAX_STRING_LENGTH = 50
    DEFAULT_DECIMAL_PLACES = 2

    # Analysis settings
    CORRELATION_THRESHOLD = 0.7
    OUTLIER_ZSCORE_THRESHOLD = 3
    MIN_SAMPLE_SIZE = 30

    # Export settings
    DEFAULT_EXPORT_FORMAT = 'csv'
    EXPORT_TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'


if __name__ == "__main__":
    # Example usage and testing
    print("Utils module loaded successfully")
    print(f"Supported formats: {Config.SUPPORTED_FILE_FORMATS}")