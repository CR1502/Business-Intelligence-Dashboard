"""
Data Processor Module for Business Intelligence Dashboard

This module handles all data loading, cleaning, validation, and filtering operations.
Implements SOLID principles with Strategy Pattern for flexible data processing.

Author: Craig
Date: December 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Optional, Any, Tuple
from abc import ABC, abstractmethod
import logging
from datetime import datetime

from utils import (
    FileValidator, DataFrameValidator, ColumnValidator,
    get_column_types, detect_date_columns, clean_currency_column,
    Config
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# STRATEGY PATTERN - Data Loading Strategies
# Follows Open/Closed Principle and Strategy Pattern
# ============================================================================

class DataLoadStrategy(ABC):
    """
    Abstract base class for data loading strategies.
    Follows Strategy Pattern - allows different loading algorithms to be selected at runtime.
    """

    @abstractmethod
    def load(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from file.

        Args:
            filepath: Path to the data file

        Returns:
            pd.DataFrame: Loaded data
        """
        pass

    @abstractmethod
    def can_handle(self, filepath: Union[str, Path]) -> bool:
        """
        Check if this strategy can handle the given file.

        Args:
            filepath: Path to check

        Returns:
            bool: True if this strategy can handle the file
        """
        pass


class CSVLoadStrategy(DataLoadStrategy):
    """
    Strategy for loading CSV files.
    Follows Single Responsibility Principle - only handles CSV loading.
    """

    def can_handle(self, filepath: Union[str, Path]) -> bool:
        """Check if file is CSV format."""
        return str(filepath).lower().endswith('.csv')

    def load(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load CSV file with automatic encoding detection.

        Args:
            filepath: Path to CSV file

        Returns:
            pd.DataFrame: Loaded data

        Raises:
            Exception: If loading fails
        """
        try:
            # Try UTF-8 first
            df = pd.read_csv(filepath, encoding='utf-8')
            logger.info(f"Successfully loaded CSV file: {filepath}")
            return df
        except UnicodeDecodeError:
            try:
                # Fallback to latin-1
                df = pd.read_csv(filepath, encoding='latin-1')
                logger.info(f"Successfully loaded CSV file with latin-1 encoding: {filepath}")
                return df
            except Exception as e:
                logger.error(f"Error loading CSV file: {e}")
                raise Exception(f"Failed to load CSV file: {str(e)}")


class ExcelLoadStrategy(DataLoadStrategy):
    """
    Strategy for loading Excel files.
    Follows Single Responsibility Principle - only handles Excel loading.
    """

    def can_handle(self, filepath: Union[str, Path]) -> bool:
        """Check if file is Excel format."""
        extension = str(filepath).lower()
        return extension.endswith('.xlsx') or extension.endswith('.xls')

    def load(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load Excel file.

        Args:
            filepath: Path to Excel file

        Returns:
            pd.DataFrame: Loaded data

        Raises:
            Exception: If loading fails
        """
        try:
            df = pd.read_excel(filepath, engine='openpyxl')
            logger.info(f"Successfully loaded Excel file: {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            raise Exception(f"Failed to load Excel file: {str(e)}")


class JSONLoadStrategy(DataLoadStrategy):
    """
    Strategy for loading JSON files.
    Follows Single Responsibility Principle - only handles JSON loading.
    """

    def can_handle(self, filepath: Union[str, Path]) -> bool:
        """Check if file is JSON format."""
        return str(filepath).lower().endswith('.json')

    def load(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            pd.DataFrame: Loaded data

        Raises:
            Exception: If loading fails
        """
        try:
            df = pd.read_json(filepath)
            logger.info(f"Successfully loaded JSON file: {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error loading JSON file: {e}")
            raise Exception(f"Failed to load JSON file: {str(e)}")


class ParquetLoadStrategy(DataLoadStrategy):
    """
    Strategy for loading Parquet files.
    Follows Single Responsibility Principle - only handles Parquet loading.
    """

    def can_handle(self, filepath: Union[str, Path]) -> bool:
        """Check if file is Parquet format."""
        return str(filepath).lower().endswith('.parquet')

    def load(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load Parquet file.

        Args:
            filepath: Path to Parquet file

        Returns:
            pd.DataFrame: Loaded data

        Raises:
            Exception: If loading fails
        """
        try:
            df = pd.read_parquet(filepath)
            logger.info(f"Successfully loaded Parquet file: {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error loading Parquet file: {e}")
            raise Exception(f"Failed to load Parquet file: {str(e)}")


# ============================================================================
# DATA LOADER CONTEXT
# Uses Strategy Pattern to select appropriate loading strategy
# ============================================================================

class DataLoader:
    """
    Context class for data loading using Strategy Pattern.
    Automatically selects the appropriate loading strategy based on file type.
    Follows Open/Closed Principle - open for extension (new strategies), closed for modification.
    """

    def __init__(self):
        """Initialize DataLoader with all available strategies."""
        self.strategies: List[DataLoadStrategy] = [
            CSVLoadStrategy(),
            ExcelLoadStrategy(),
            JSONLoadStrategy(),
            ParquetLoadStrategy()
        ]
        self.file_validator = FileValidator()

    def load_data(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load data using appropriate strategy.

        Args:
            filepath: Path to data file

        Returns:
            pd.DataFrame: Loaded data

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
            Exception: If loading fails
        """
        # Validate file
        self.file_validator.validate(filepath)

        # Find appropriate strategy
        for strategy in self.strategies:
            if strategy.can_handle(filepath):
                df = strategy.load(filepath)
                logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
                return df

        # No strategy found
        raise ValueError(f"No loading strategy available for file: {filepath}")

    def add_strategy(self, strategy: DataLoadStrategy) -> None:
        """
        Add a new loading strategy.
        Follows Open/Closed Principle - extend functionality without modifying existing code.

        Args:
            strategy: New loading strategy to add
        """
        self.strategies.append(strategy)
        logger.info(f"Added new loading strategy: {strategy.__class__.__name__}")


# ============================================================================
# DATA CLEANING
# Follows Single Responsibility Principle
# ============================================================================

class DataCleaner:
    """
    Handles data cleaning operations.
    Follows Single Responsibility Principle - only responsible for cleaning data.
    """

    @staticmethod
    def handle_missing_values(df: pd.DataFrame, strategy: str = 'none') -> pd.DataFrame:
        """
        Handle missing values in DataFrame.

        Args:
            df: DataFrame to clean
            strategy: Strategy for handling missing values
                     'none' - do nothing
                     'drop' - drop rows with any missing values
                     'fill_mean' - fill numerical columns with mean
                     'fill_median' - fill numerical columns with median
                     'fill_mode' - fill categorical columns with mode

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        if strategy == 'none':
            return df.copy()

        df_cleaned = df.copy()

        if strategy == 'drop':
            df_cleaned = df_cleaned.dropna()
            logger.info(f"Dropped rows with missing values. Remaining rows: {len(df_cleaned)}")

        elif strategy == 'fill_mean':
            numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
            logger.info(f"Filled missing values with mean for {len(numerical_cols)} columns")

        elif strategy == 'fill_median':
            numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
            logger.info(f"Filled missing values with median for {len(numerical_cols)} columns")

        elif strategy == 'fill_mode':
            for col in df_cleaned.columns:
                if df_cleaned[col].dtype == 'object':
                    mode_value = df_cleaned[col].mode()
                    if len(mode_value) > 0:
                        df_cleaned[col].fillna(mode_value[0], inplace=True)
            logger.info("Filled missing values with mode for categorical columns")

        return df_cleaned

    @staticmethod
    def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
        """
        Automatically convert data types (dates, currencies, etc.).

        Args:
            df: DataFrame to convert

        Returns:
            pd.DataFrame: DataFrame with converted types
        """
        df_converted = df.copy()

        # Detect and convert date columns
        date_columns = detect_date_columns(df_converted)
        for col in date_columns:
            try:
                df_converted[col] = pd.to_datetime(df_converted[col], errors='coerce')
                logger.info(f"Converted column '{col}' to datetime")
            except Exception as e:
                logger.warning(f"Could not convert '{col}' to datetime: {e}")

        # Detect and convert currency columns
        for col in df_converted.select_dtypes(include=['object']).columns:
            # Check if column contains currency symbols
            sample = df_converted[col].dropna().head(10).astype(str)
            if any(any(symbol in str(val) for symbol in ['$', '€', '£', '¥']) for val in sample):
                try:
                    df_converted[col] = clean_currency_column(df_converted[col])
                    logger.info(f"Converted column '{col}' from currency to numeric")
                except Exception as e:
                    logger.warning(f"Could not convert '{col}' from currency: {e}")

        # Convert boolean strings to actual booleans
        for col in df_converted.select_dtypes(include=['object']).columns:
            unique_values = df_converted[col].dropna().unique()
            if len(unique_values) <= 2 and all(
                    str(v).upper() in ['TRUE', 'FALSE', 'YES', 'NO', '0', '1'] for v in unique_values):
                try:
                    df_converted[col] = df_converted[col].map({
                        'TRUE': True, 'FALSE': False,
                        'YES': True, 'NO': False,
                        'True': True, 'False': False,
                        'true': True, 'false': False,
                        '1': True, '0': False
                    })
                    logger.info(f"Converted column '{col}' to boolean")
                except Exception as e:
                    logger.warning(f"Could not convert '{col}' to boolean: {e}")

        return df_converted

    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from DataFrame.

        Args:
            df: DataFrame to clean

        Returns:
            pd.DataFrame: DataFrame without duplicates
        """
        initial_rows = len(df)
        df_cleaned = df.drop_duplicates()
        removed_rows = initial_rows - len(df_cleaned)

        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} duplicate rows")

        return df_cleaned

    @staticmethod
    def handle_outliers(df: pd.DataFrame, columns: List[str], method: str = 'zscore',
                        threshold: float = 3.0) -> pd.DataFrame:
        """
        Handle outliers in numerical columns.

        Args:
            df: DataFrame to process
            columns: List of columns to check for outliers
            method: Method for outlier detection ('zscore' or 'iqr')
            threshold: Threshold for outlier detection

        Returns:
            pd.DataFrame: DataFrame with outliers handled
        """
        df_cleaned = df.copy()

        for col in columns:
            if col not in df_cleaned.columns:
                continue

            if not pd.api.types.is_numeric_dtype(df_cleaned[col]):
                continue

            if method == 'zscore':
                # Z-score method
                z_scores = np.abs((df_cleaned[col] - df_cleaned[col].mean()) / df_cleaned[col].std())
                df_cleaned = df_cleaned[z_scores < threshold]

            elif method == 'iqr':
                # IQR method
                Q1 = df_cleaned[col].quantile(0.25)
                Q3 = df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]

        removed_rows = len(df) - len(df_cleaned)
        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} outlier rows")

        return df_cleaned


# ============================================================================
# DATA PROFILER
# Generates comprehensive statistics about the dataset
# ============================================================================

class DataProfiler:
    """
    Generates comprehensive data profiling statistics.
    Follows Single Responsibility Principle - only responsible for profiling.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize profiler with DataFrame.

        Args:
            df: DataFrame to profile
        """
        self.df = df
        self.validator = DataFrameValidator()
        self.validator.validate(df)

    def get_basic_info(self) -> Dict[str, Any]:
        """
        Get basic information about the dataset.

        Returns:
            Dict with shape, columns, data types, and memory usage
        """
        return {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'column_names': self.df.columns.tolist(),
            'data_types': self.df.dtypes.to_dict(),
            'memory_usage': f"{self.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
        }

    def get_missing_values_report(self) -> pd.DataFrame:
        """
        Generate report on missing values.

        Returns:
            DataFrame with missing value statistics per column
        """
        missing_data = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': self.df.isnull().sum().values,
            'Missing_Percentage': (self.df.isnull().sum().values / len(self.df) * 100).round(2)
        })

        return missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

    def get_numerical_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for numerical columns.

        Returns:
            DataFrame with descriptive statistics
        """
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns

        if len(numerical_cols) == 0:
            return pd.DataFrame()

        return self.df[numerical_cols].describe()

    def get_categorical_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get summary statistics for categorical columns.

        Returns:
            Dict with statistics for each categorical column
        """
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns

        summary = {}
        for col in categorical_cols:
            # Get value counts, dropping NaN values
            value_counts = self.df[col].value_counts()

            # Safely get mode
            mode_values = self.df[col].mode()
            top_value = mode_values.iloc[0] if len(mode_values) > 0 and not mode_values.empty else None

            # Safely get top frequency
            top_freq = value_counts.iloc[0] if len(value_counts) > 0 else 0

            summary[col] = {
                'unique_count': self.df[col].nunique(),
                'top_value': top_value,
                'top_value_frequency': top_freq,
                'value_counts': value_counts.head(10).to_dict()
            }

        return summary

    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Get correlation matrix for numerical columns.

        Returns:
            Correlation matrix DataFrame
        """
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns

        if len(numerical_cols) < 2:
            return pd.DataFrame()

        return self.df[numerical_cols].corr()

    def get_full_profile(self) -> Dict[str, Any]:
        """
        Get comprehensive data profile.

        Returns:
            Dict with all profiling information
        """
        return {
            'basic_info': self.get_basic_info(),
            'missing_values': self.get_missing_values_report(),
            'numerical_summary': self.get_numerical_summary(),
            'categorical_summary': self.get_categorical_summary(),
            'correlation_matrix': self.get_correlation_matrix()
        }


# ============================================================================
# DATA FILTER
# Handles interactive filtering operations
# ============================================================================

class DataFilter:
    """
    Handles data filtering operations.
    Follows Single Responsibility Principle - only responsible for filtering.
    """

    @staticmethod
    def filter_numerical(df: pd.DataFrame, column: str, min_val: Optional[float] = None,
                         max_val: Optional[float] = None) -> pd.DataFrame:
        """
        Filter DataFrame by numerical column range.

        Args:
            df: DataFrame to filter
            column: Column name to filter
            min_val: Minimum value (inclusive)
            max_val: Maximum value (inclusive)

        Returns:
            Filtered DataFrame
        """
        ColumnValidator().validate(df, column)

        filtered_df = df.copy()

        if min_val is not None:
            filtered_df = filtered_df[filtered_df[column] >= min_val]

        if max_val is not None:
            filtered_df = filtered_df[filtered_df[column] <= max_val]

        logger.info(f"Filtered by {column}: {len(filtered_df)} rows remaining")
        return filtered_df

    @staticmethod
    def filter_categorical(df: pd.DataFrame, column: str, values: List[Any]) -> pd.DataFrame:
        """
        Filter DataFrame by categorical column values.

        Args:
            df: DataFrame to filter
            column: Column name to filter
            values: List of values to keep

        Returns:
            Filtered DataFrame
        """
        ColumnValidator().validate(df, column)

        if not values:
            return df.copy()

        filtered_df = df[df[column].isin(values)]
        logger.info(f"Filtered by {column}: {len(filtered_df)} rows remaining")
        return filtered_df

    @staticmethod
    def filter_date_range(df: pd.DataFrame, column: str, start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Filter DataFrame by date range.

        Args:
            df: DataFrame to filter
            column: Date column name
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            Filtered DataFrame
        """
        ColumnValidator().validate(df, column)

        filtered_df = df.copy()

        # Ensure column is datetime
        if not pd.api.types.is_datetime64_any_dtype(filtered_df[column]):
            filtered_df[column] = pd.to_datetime(filtered_df[column], errors='coerce')

        if start_date is not None:
            filtered_df = filtered_df[filtered_df[column] >= start_date]

        if end_date is not None:
            filtered_df = filtered_df[filtered_df[column] <= end_date]

        logger.info(f"Filtered by date range on {column}: {len(filtered_df)} rows remaining")
        return filtered_df

    @staticmethod
    def apply_multiple_filters(df: pd.DataFrame, filters: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Apply multiple filters sequentially.

        Args:
            df: DataFrame to filter
            filters: List of filter dictionaries with keys:
                    - 'type': 'numerical', 'categorical', or 'date'
                    - 'column': column name
                    - other keys depending on filter type

        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()

        for filter_config in filters:
            filter_type = filter_config.get('type')
            column = filter_config.get('column')

            if filter_type == 'numerical':
                filtered_df = DataFilter.filter_numerical(
                    filtered_df,
                    column,
                    filter_config.get('min_val'),
                    filter_config.get('max_val')
                )

            elif filter_type == 'categorical':
                filtered_df = DataFilter.filter_categorical(
                    filtered_df,
                    column,
                    filter_config.get('values', [])
                )

            elif filter_type == 'date':
                filtered_df = DataFilter.filter_date_range(
                    filtered_df,
                    column,
                    filter_config.get('start_date'),
                    filter_config.get('end_date')
                )

        return filtered_df


# ============================================================================
# MAIN DATA PROCESSOR CLASS
# Facade pattern - provides simple interface to complex subsystems
# ============================================================================

class DataProcessor:
    """
    Main data processor class using Facade pattern.
    Provides simple interface to complex data loading, cleaning, and filtering operations.
    Follows Dependency Inversion Principle - depends on abstractions, not concrete implementations.
    """

    def __init__(self):
        """Initialize DataProcessor with all components."""
        self.loader = DataLoader()
        self.cleaner = DataCleaner()
        self.filter = DataFilter()
        self.current_df: Optional[pd.DataFrame] = None
        self.original_df: Optional[pd.DataFrame] = None
        self.profiler: Optional[DataProfiler] = None

    def load_and_prepare_data(self, filepath: Union[str, Path],
                              clean: bool = True,
                              remove_duplicates: bool = True) -> pd.DataFrame:
        """
        Load and prepare data with automatic cleaning.

        Args:
            filepath: Path to data file
            clean: Whether to apply automatic type conversion
            remove_duplicates: Whether to remove duplicate rows

        Returns:
            Prepared DataFrame
        """
        # Load data
        df = self.loader.load_data(filepath)
        self.original_df = df.copy()

        # Clean data
        if clean:
            df = self.cleaner.convert_data_types(df)

        if remove_duplicates:
            df = self.cleaner.remove_duplicates(df)

        self.current_df = df
        self.profiler = DataProfiler(df)

        logger.info("Data loaded and prepared successfully")
        return df

    def get_data_profile(self) -> Dict[str, Any]:
        """
        Get comprehensive data profile.

        Returns:
            Dict with profiling information
        """
        if self.profiler is None:
            raise ValueError("No data loaded. Call load_and_prepare_data first.")

        return self.profiler.get_full_profile()

    def apply_filters(self, filters: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Apply filters to current data.

        Args:
            filters: List of filter configurations

        Returns:
            Filtered DataFrame
        """
        if self.current_df is None:
            raise ValueError("No data loaded. Call load_and_prepare_data first.")

        return self.filter.apply_multiple_filters(self.current_df, filters)

    def reset_to_original(self) -> pd.DataFrame:
        """
        Reset current data to original loaded data.

        Returns:
            Original DataFrame
        """
        if self.original_df is None:
            raise ValueError("No data loaded. Call load_and_prepare_data first.")

        self.current_df = self.original_df.copy()
        return self.current_df

    def get_column_info(self) -> Dict[str, List[str]]:
        """
        Get categorized column information.

        Returns:
            Dict with numerical, categorical, and datetime columns
        """
        if self.current_df is None:
            raise ValueError("No data loaded. Call load_and_prepare_data first.")

        return get_column_types(self.current_df)


if __name__ == "__main__":
    # Example usage
    print("DataProcessor module loaded successfully")

    # Demonstrate Strategy Pattern
    processor = DataProcessor()
    print(f"Available strategies: {len(processor.loader.strategies)}")