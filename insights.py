"""
Insights Module for Business Intelligence Dashboard

This module handles automated insight generation from data.
Uses Strategy Pattern for different types of insights.

Author: Craig
Date: December 2024
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Optional, Any, Tuple
from abc import ABC, abstractmethod
import logging
from datetime import datetime, timedelta

from utils import (
    DataFrameValidator, ColumnValidator,
    format_number, format_percentage, safe_divide,
    get_column_types
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# STRATEGY PATTERN - Insight Strategies
# Follows Open/Closed Principle and Strategy Pattern
# ============================================================================

class InsightStrategy(ABC):
    """
    Abstract base class for insight generation strategies.
    Follows Strategy Pattern - allows different insight algorithms.
    """

    @abstractmethod
    def generate(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate insights from data.

        Args:
            df: DataFrame to analyze
            **kwargs: Additional parameters for insight generation

        Returns:
            Dict containing insight information
        """
        pass

    @abstractmethod
    def get_insight_type(self) -> str:
        """
        Get the type of insight this strategy generates.

        Returns:
            str: Insight type name
        """
        pass


# ============================================================================
# TOP/BOTTOM PERFORMERS INSIGHTS
# ============================================================================

class TopBottomPerformers(InsightStrategy):
    """
    Identify top and bottom performers in the data.
    Follows Single Responsibility Principle - only handles top/bottom analysis.
    """

    def get_insight_type(self) -> str:
        """Get insight type."""
        return "top_bottom_performers"

    def generate(self, df: pd.DataFrame,
                 column: str,
                 group_by: Optional[str] = None,
                 top_n: int = 5,
                 bottom_n: int = 5,
                 aggregation: str = 'sum',
                 **kwargs) -> Dict[str, Any]:
        """
        Generate top and bottom performer insights.

        Args:
            df: DataFrame to analyze
            column: Column to analyze for performance
            group_by: Optional column to group by
            top_n: Number of top performers to identify
            bottom_n: Number of bottom performers to identify
            aggregation: Aggregation method if group_by is used
            **kwargs: Additional parameters

        Returns:
            Dict with top and bottom performers
        """
        # Validate inputs
        DataFrameValidator().validate(df)
        ColumnValidator().validate(df, column)

        if group_by:
            ColumnValidator().validate(df, group_by)

            # Aggregate by group
            if aggregation == 'sum':
                data = df.groupby(group_by)[column].sum().sort_values(ascending=False)
            elif aggregation == 'mean':
                data = df.groupby(group_by)[column].mean().sort_values(ascending=False)
            elif aggregation == 'count':
                data = df.groupby(group_by)[column].count().sort_values(ascending=False)
            elif aggregation == 'median':
                data = df.groupby(group_by)[column].median().sort_values(ascending=False)
            else:
                data = df.groupby(group_by)[column].sum().sort_values(ascending=False)
        else:
            # Direct analysis on column
            data = df[column].sort_values(ascending=False)

        # Get top and bottom performers
        top_performers = data.head(top_n)
        bottom_performers = data.tail(bottom_n).sort_values(ascending=True)

        # Calculate statistics
        total = data.sum()
        top_contribution = safe_divide(top_performers.sum(), total) if total != 0 else 0
        bottom_contribution = safe_divide(bottom_performers.sum(), total) if total != 0 else 0

        insight = {
            'type': self.get_insight_type(),
            'column': column,
            'group_by': group_by,
            'aggregation': aggregation if group_by else 'direct',
            'top_performers': {
                'data': top_performers.to_dict(),
                'count': len(top_performers),
                'total_value': top_performers.sum(),
                'contribution_percentage': top_contribution
            },
            'bottom_performers': {
                'data': bottom_performers.to_dict(),
                'count': len(bottom_performers),
                'total_value': bottom_performers.sum(),
                'contribution_percentage': bottom_contribution
            },
            'summary': self._generate_summary(
                column, group_by, top_performers, bottom_performers,
                top_contribution, bottom_contribution
            )
        }

        logger.info(f"Generated top/bottom performers insight for {column}")
        return insight

    def _generate_summary(self, column: str, group_by: Optional[str],
                          top: pd.Series, bottom: pd.Series,
                          top_contrib: float, bottom_contrib: float) -> str:
        """Generate human-readable summary."""
        if group_by:
            top_name = top.index[0] if len(top) > 0 else "N/A"
            bottom_name = bottom.index[0] if len(bottom) > 0 else "N/A"

            summary = f"Top performer in {column}: '{top_name}' with {format_number(top.iloc[0])}. "
            summary += f"Bottom performer: '{bottom_name}' with {format_number(bottom.iloc[0])}. "
            summary += f"Top {len(top)} performers contribute {format_percentage(top_contrib)} of total."
        else:
            summary = f"Highest value in {column}: {format_number(top.iloc[0])}. "
            summary += f"Lowest value: {format_number(bottom.iloc[0])}. "
            summary += f"Range: {format_number(top.iloc[0] - bottom.iloc[0])}"

        return summary


# ============================================================================
# TREND ANALYSIS INSIGHTS
# ============================================================================

class TrendAnalysis(InsightStrategy):
    """
    Analyze trends in time series data.
    Follows Single Responsibility Principle - only handles trend analysis.
    """

    def get_insight_type(self) -> str:
        """Get insight type."""
        return "trend_analysis"

    def generate(self, df: pd.DataFrame,
                 date_column: str,
                 value_column: str,
                 period: str = 'overall',
                 **kwargs) -> Dict[str, Any]:
        """
        Generate trend analysis insights.

        Args:
            df: DataFrame to analyze
            date_column: Column containing dates
            value_column: Column containing values
            period: Analysis period ('overall', 'monthly', 'weekly', 'daily')
            **kwargs: Additional parameters

        Returns:
            Dict with trend insights
        """
        # Validate inputs
        DataFrameValidator().validate(df)
        ColumnValidator().validate(df, [date_column, value_column])

        # Prepare data
        df_trend = df[[date_column, value_column]].copy()

        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_trend[date_column]):
            df_trend[date_column] = pd.to_datetime(df_trend[date_column], errors='coerce')

        # Remove NaN values
        df_trend = df_trend.dropna()

        if len(df_trend) < 2:
            return {
                'type': self.get_insight_type(),
                'error': 'Insufficient data for trend analysis',
                'summary': 'Not enough data points to analyze trends.'
            }

        # Sort by date
        df_trend = df_trend.sort_values(date_column)

        # Calculate trend metrics
        first_value = df_trend[value_column].iloc[0]
        last_value = df_trend[value_column].iloc[-1]
        change = last_value - first_value
        change_pct = safe_divide(change, first_value)

        # Determine trend direction
        if change > 0:
            trend_direction = 'increasing'
        elif change < 0:
            trend_direction = 'decreasing'
        else:
            trend_direction = 'stable'

        # Calculate statistics
        mean_value = df_trend[value_column].mean()
        median_value = df_trend[value_column].median()
        std_value = df_trend[value_column].std()

        # Calculate growth rate (if applicable)
        growth_rate = self._calculate_growth_rate(df_trend, date_column, value_column)

        # Detect volatility
        volatility = self._calculate_volatility(df_trend[value_column])

        insight = {
            'type': self.get_insight_type(),
            'date_column': date_column,
            'value_column': value_column,
            'period': period,
            'trend_direction': trend_direction,
            'metrics': {
                'first_value': first_value,
                'last_value': last_value,
                'absolute_change': change,
                'percentage_change': change_pct,
                'mean': mean_value,
                'median': median_value,
                'std_deviation': std_value,
                'growth_rate': growth_rate,
                'volatility': volatility
            },
            'date_range': {
                'start': df_trend[date_column].min().strftime('%Y-%m-%d'),
                'end': df_trend[date_column].max().strftime('%Y-%m-%d'),
                'days': (df_trend[date_column].max() - df_trend[date_column].min()).days
            },
            'summary': self._generate_summary(
                value_column, trend_direction, change, change_pct, volatility
            )
        }

        logger.info(f"Generated trend analysis insight for {value_column}")
        return insight

    def _calculate_growth_rate(self, df: pd.DataFrame,
                               date_col: str, value_col: str) -> Optional[float]:
        """Calculate average growth rate."""
        try:
            # Simple linear regression for growth rate
            x = (df[date_col] - df[date_col].min()).dt.days.values
            y = df[value_col].values

            if len(x) < 2:
                return None

            # Calculate slope
            slope = np.polyfit(x, y, 1)[0]
            return slope
        except Exception:
            return None

    def _calculate_volatility(self, series: pd.Series) -> str:
        """Calculate volatility level."""
        if len(series) < 2:
            return 'unknown'

        # Use coefficient of variation
        cv = safe_divide(series.std(), series.mean())

        if cv < 0.1:
            return 'low'
        elif cv < 0.3:
            return 'moderate'
        else:
            return 'high'

    def _generate_summary(self, column: str, direction: str,
                          change: float, change_pct: float, volatility: str) -> str:
        """Generate human-readable summary."""
        summary = f"{column} shows a {direction} trend with "
        summary += f"{format_percentage(abs(change_pct))} {'increase' if change > 0 else 'decrease'}. "
        summary += f"Absolute change: {format_number(change)}. "
        summary += f"Volatility: {volatility}."
        return summary


# ============================================================================
# ANOMALY DETECTION INSIGHTS
# ============================================================================

class AnomalyDetection(InsightStrategy):
    """
    Detect anomalies and outliers in data.
    Follows Single Responsibility Principle - only handles anomaly detection.
    """

    def get_insight_type(self) -> str:
        """Get insight type."""
        return "anomaly_detection"

    def generate(self, df: pd.DataFrame,
                 column: str,
                 method: str = 'zscore',
                 threshold: float = 3.0,
                 **kwargs) -> Dict[str, Any]:
        """
        Generate anomaly detection insights.

        Args:
            df: DataFrame to analyze
            column: Column to analyze for anomalies
            method: Detection method ('zscore' or 'iqr')
            threshold: Threshold for anomaly detection
            **kwargs: Additional parameters

        Returns:
            Dict with anomaly insights
        """
        # Validate inputs
        DataFrameValidator().validate(df)
        ColumnValidator().validate(df, column)

        # Check if column is numerical
        if not pd.api.types.is_numeric_dtype(df[column]):
            return {
                'type': self.get_insight_type(),
                'error': f'Column {column} is not numerical',
                'summary': f'Cannot detect anomalies in non-numerical column {column}.'
            }

        # Remove NaN values
        data = df[column].dropna()

        if len(data) < 3:
            return {
                'type': self.get_insight_type(),
                'error': 'Insufficient data',
                'summary': 'Not enough data points to detect anomalies.'
            }

        # Detect anomalies
        if method == 'zscore':
            anomalies_mask = self._detect_zscore(data, threshold)
        elif method == 'iqr':
            anomalies_mask = self._detect_iqr(data, threshold)
        else:
            raise ValueError(f"Unsupported method: {method}")

        anomalies = data[anomalies_mask]

        # Calculate statistics
        total_points = len(data)
        anomaly_count = len(anomalies)
        anomaly_percentage = safe_divide(anomaly_count, total_points)

        insight = {
            'type': self.get_insight_type(),
            'column': column,
            'method': method,
            'threshold': threshold,
            'statistics': {
                'total_points': total_points,
                'anomaly_count': anomaly_count,
                'anomaly_percentage': anomaly_percentage,
                'mean': data.mean(),
                'median': data.median(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max()
            },
            'anomalies': {
                'values': anomalies.tolist()[:20],  # Limit to first 20
                'max_anomaly': anomalies.max() if len(anomalies) > 0 else None,
                'min_anomaly': anomalies.min() if len(anomalies) > 0 else None
            },
            'summary': self._generate_summary(
                column, method, anomaly_count, anomaly_percentage,
                anomalies.max() if len(anomalies) > 0 else None,
                anomalies.min() if len(anomalies) > 0 else None
            )
        }

        logger.info(f"Generated anomaly detection insight for {column}")
        return insight

    def _detect_zscore(self, series: pd.Series, threshold: float) -> pd.Series:
        """Detect anomalies using Z-score method."""
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold

    def _detect_iqr(self, series: pd.Series, threshold: float) -> pd.Series:
        """Detect anomalies using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (series < lower_bound) | (series > upper_bound)

    def _generate_summary(self, column: str, method: str,
                          count: int, percentage: float,
                          max_anomaly: Optional[float],
                          min_anomaly: Optional[float]) -> str:
        """Generate human-readable summary."""
        if count == 0:
            return f"No anomalies detected in {column} using {method} method."

        summary = f"Detected {count} anomalies ({format_percentage(percentage)}) in {column}. "

        if max_anomaly and min_anomaly:
            summary += f"Range of anomalies: {format_number(min_anomaly)} to {format_number(max_anomaly)}."

        return summary


# ============================================================================
# DISTRIBUTION INSIGHTS
# ============================================================================

class DistributionInsights(InsightStrategy):
    """
    Analyze data distribution characteristics.
    Follows Single Responsibility Principle - only handles distribution analysis.
    """

    def get_insight_type(self) -> str:
        """Get insight type."""
        return "distribution_insights"

    def generate(self, df: pd.DataFrame,
                 column: str,
                 **kwargs) -> Dict[str, Any]:
        """
        Generate distribution insights.

        Args:
            df: DataFrame to analyze
            column: Column to analyze
            **kwargs: Additional parameters

        Returns:
            Dict with distribution insights
        """
        # Validate inputs
        DataFrameValidator().validate(df)
        ColumnValidator().validate(df, column)

        # Check if column is numerical
        if not pd.api.types.is_numeric_dtype(df[column]):
            # For categorical columns
            return self._categorical_distribution(df, column)
        else:
            # For numerical columns
            return self._numerical_distribution(df, column)

    def _numerical_distribution(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Analyze numerical distribution."""
        data = df[column].dropna()

        if len(data) == 0:
            return {
                'type': self.get_insight_type(),
                'error': 'No valid data',
                'summary': f'No valid data in column {column}.'
            }

        # Calculate statistics
        statistics = {
            'count': len(data),
            'mean': data.mean(),
            'median': data.median(),
            'mode': data.mode()[0] if len(data.mode()) > 0 else None,
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'range': data.max() - data.min(),
            'q1': data.quantile(0.25),
            'q3': data.quantile(0.75),
            'iqr': data.quantile(0.75) - data.quantile(0.25),
            'skewness': data.skew(),
            'kurtosis': data.kurtosis()
        }

        # Determine distribution shape
        shape = self._determine_shape(statistics['skewness'], statistics['kurtosis'])

        insight = {
            'type': self.get_insight_type(),
            'column': column,
            'data_type': 'numerical',
            'statistics': statistics,
            'distribution_shape': shape,
            'summary': self._generate_numerical_summary(column, statistics, shape)
        }

        logger.info(f"Generated distribution insight for {column}")
        return insight

    def _categorical_distribution(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Analyze categorical distribution."""
        data = df[column].dropna()

        if len(data) == 0:
            return {
                'type': self.get_insight_type(),
                'error': 'No valid data',
                'summary': f'No valid data in column {column}.'
            }

        # Calculate statistics
        value_counts = data.value_counts()

        statistics = {
            'count': len(data),
            'unique_values': data.nunique(),
            'most_common': value_counts.index[0],
            'most_common_count': value_counts.iloc[0],
            'most_common_percentage': safe_divide(value_counts.iloc[0], len(data)),
            'least_common': value_counts.index[-1],
            'least_common_count': value_counts.iloc[-1]
        }

        insight = {
            'type': self.get_insight_type(),
            'column': column,
            'data_type': 'categorical',
            'statistics': statistics,
            'value_counts': value_counts.head(10).to_dict(),
            'summary': self._generate_categorical_summary(column, statistics)
        }

        logger.info(f"Generated distribution insight for {column}")
        return insight

    def _determine_shape(self, skewness: float, kurtosis: float) -> str:
        """Determine distribution shape from skewness and kurtosis."""
        if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
            return 'approximately normal'
        elif skewness > 0.5:
            return 'right-skewed (positive skew)'
        elif skewness < -0.5:
            return 'left-skewed (negative skew)'
        elif kurtosis > 1:
            return 'heavy-tailed (leptokurtic)'
        elif kurtosis < -1:
            return 'light-tailed (platykurtic)'
        else:
            return 'mixed characteristics'

    def _generate_numerical_summary(self, column: str,
                                    stats: Dict, shape: str) -> str:
        """Generate summary for numerical distribution."""
        summary = f"{column} has a {shape} distribution. "
        summary += f"Mean: {format_number(stats['mean'])}, "
        summary += f"Median: {format_number(stats['median'])}, "
        summary += f"Std Dev: {format_number(stats['std'])}. "
        summary += f"Range: {format_number(stats['min'])} to {format_number(stats['max'])}."
        return summary

    def _generate_categorical_summary(self, column: str, stats: Dict) -> str:
        """Generate summary for categorical distribution."""
        summary = f"{column} has {stats['unique_values']} unique values. "
        summary += f"Most common: '{stats['most_common']}' "
        summary += f"({format_percentage(stats['most_common_percentage'])})."
        return summary


# ============================================================================
# CORRELATION INSIGHTS
# ============================================================================

class CorrelationInsights(InsightStrategy):
    """
    Identify strong correlations between variables.
    Follows Single Responsibility Principle - only handles correlation analysis.
    """

    def get_insight_type(self) -> str:
        """Get insight type."""
        return "correlation_insights"

    def generate(self, df: pd.DataFrame,
                 columns: Optional[List[str]] = None,
                 threshold: float = 0.7,
                 method: str = 'pearson',
                 **kwargs) -> Dict[str, Any]:
        """
        Generate correlation insights.

        Args:
            df: DataFrame to analyze
            columns: Optional list of columns to analyze
            threshold: Correlation threshold for strong correlations
            method: Correlation method ('pearson', 'spearman', 'kendall')
            **kwargs: Additional parameters

        Returns:
            Dict with correlation insights
        """
        # Validate inputs
        DataFrameValidator().validate(df)

        # Select numerical columns
        if columns:
            ColumnValidator().validate(df, columns)
            df_corr = df[columns].select_dtypes(include=[np.number])
        else:
            df_corr = df.select_dtypes(include=[np.number])

        if df_corr.shape[1] < 2:
            return {
                'type': self.get_insight_type(),
                'error': 'Insufficient numerical columns',
                'summary': 'Need at least 2 numerical columns for correlation analysis.'
            }

        # Calculate correlation matrix
        corr_matrix = df_corr.corr(method=method)

        # Find strong correlations
        strong_correlations = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]

                if abs(corr_value) >= threshold:
                    strong_correlations.append({
                        'variable1': corr_matrix.columns[i],
                        'variable2': corr_matrix.columns[j],
                        'correlation': corr_value,
                        'strength': self._classify_strength(abs(corr_value)),
                        'direction': 'positive' if corr_value > 0 else 'negative'
                    })

        # Sort by absolute correlation value
        strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)

        insight = {
            'type': self.get_insight_type(),
            'method': method,
            'threshold': threshold,
            'total_pairs_analyzed': len(corr_matrix.columns) * (len(corr_matrix.columns) - 1) // 2,
            'strong_correlations_found': len(strong_correlations),
            'correlations': strong_correlations[:10],  # Top 10
            'summary': self._generate_summary(strong_correlations, threshold)
        }

        logger.info(f"Generated correlation insights with {len(strong_correlations)} strong correlations")
        return insight

    def _classify_strength(self, abs_corr: float) -> str:
        """Classify correlation strength."""
        if abs_corr >= 0.9:
            return 'very strong'
        elif abs_corr >= 0.7:
            return 'strong'
        elif abs_corr >= 0.5:
            return 'moderate'
        elif abs_corr >= 0.3:
            return 'weak'
        else:
            return 'very weak'

    def _generate_summary(self, correlations: List[Dict], threshold: float) -> str:
        """Generate human-readable summary."""
        if len(correlations) == 0:
            return f"No strong correlations (threshold: {threshold}) found."

        top = correlations[0]
        summary = f"Found {len(correlations)} strong correlations. "
        summary += f"Strongest: {top['variable1']} and {top['variable2']} "
        summary += f"({top['direction']}, {format_number(top['correlation'])})."

        return summary


# ============================================================================
# INSIGHT MANAGER
# Uses Strategy Pattern to manage different insight types
# ============================================================================

class InsightManager:
    """
    Manager class for insights using Strategy Pattern.
    Follows Open/Closed Principle - open for extension, closed for modification.
    """

    def __init__(self):
        """Initialize InsightManager with all available strategies."""
        self.strategies: Dict[str, InsightStrategy] = {
            'top_bottom': TopBottomPerformers(),
            'trend': TrendAnalysis(),
            'anomaly': AnomalyDetection(),
            'distribution': DistributionInsights(),
            'correlation': CorrelationInsights()
        }

    def generate_insight(self, insight_type: str, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate insight using specified strategy.

        Args:
            insight_type: Type of insight to generate
            df: DataFrame to analyze
            **kwargs: Parameters specific to insight type

        Returns:
            Dict with insight information

        Raises:
            ValueError: If insight type is not supported
        """
        if insight_type not in self.strategies:
            raise ValueError(
                f"Unsupported insight type: {insight_type}. "
                f"Available types: {list(self.strategies.keys())}"
            )

        strategy = self.strategies[insight_type]
        return strategy.generate(df, **kwargs)

    def generate_all_insights(self, df: pd.DataFrame,
                              config: Optional[Dict[str, Dict]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Generate all available insights.

        Args:
            df: DataFrame to analyze
            config: Optional configuration for each insight type

        Returns:
            Dict with all insights
        """
        all_insights = {}

        # Get column types
        column_types = get_column_types(df)

        # Generate insights based on available data
        try:
            # Top/Bottom performers (if numerical columns exist)
            if len(column_types['numerical']) > 0:
                col = column_types['numerical'][0]
                params = config.get('top_bottom', {}) if config else {}
                all_insights['top_bottom'] = self.generate_insight(
                    'top_bottom', df, column=col, **params
                )
        except Exception as e:
            logger.warning(f"Could not generate top/bottom insight: {e}")

        try:
            # Distribution insights
            if len(column_types['numerical']) > 0:
                col = column_types['numerical'][0]
                params = config.get('distribution', {}) if config else {}
                all_insights['distribution'] = self.generate_insight(
                    'distribution', df, column=col, **params
                )
        except Exception as e:
            logger.warning(f"Could not generate distribution insight: {e}")

        try:
            # Anomaly detection
            if len(column_types['numerical']) > 0:
                col = column_types['numerical'][0]
                params = config.get('anomaly', {}) if config else {}
                all_insights['anomaly'] = self.generate_insight(
                    'anomaly', df, column=col, **params
                )
        except Exception as e:
            logger.warning(f"Could not generate anomaly insight: {e}")

        try:
            # Correlation insights
            if len(column_types['numerical']) >= 2:
                params = config.get('correlation', {}) if config else {}
                all_insights['correlation'] = self.generate_insight(
                    'correlation', df, **params
                )
        except Exception as e:
            logger.warning(f"Could not generate correlation insight: {e}")

        try:
            # Trend analysis (if datetime columns exist)
            if len(column_types['datetime']) > 0 and len(column_types['numerical']) > 0:
                date_col = column_types['datetime'][0]
                value_col = column_types['numerical'][0]
                params = config.get('trend', {}) if config else {}
                all_insights['trend'] = self.generate_insight(
                    'trend', df, date_column=date_col, value_column=value_col, **params
                )
        except Exception as e:
            logger.warning(f"Could not generate trend insight: {e}")

        return all_insights

    def add_strategy(self, name: str, strategy: InsightStrategy) -> None:
        """
        Add new insight strategy.
        Follows Open/Closed Principle - extend functionality without modifying existing code.

        Args:
            name: Name for the strategy
            strategy: Insight strategy instance
        """
        self.strategies[name] = strategy
        logger.info(f"Added new insight strategy: {name}")

    def get_available_insights(self) -> List[str]:
        """
        Get list of available insight types.

        Returns:
            List of insight type names
        """
        return list(self.strategies.keys())

    def format_insight_report(self, insights: Dict[str, Dict[str, Any]]) -> str:
        """
        Format insights into a readable report.

        Args:
            insights: Dict of insights from generate_all_insights

        Returns:
            Formatted string report
        """
        report = "=" * 80 + "\n"
        report += "AUTOMATED INSIGHTS REPORT\n"
        report += "=" * 80 + "\n\n"

        for insight_name, insight_data in insights.items():
            report += f"\n{insight_name.upper().replace('_', ' ')}\n"
            report += "-" * 80 + "\n"

            if 'error' in insight_data:
                report += f"Error: {insight_data['error']}\n"
            elif 'summary' in insight_data:
                report += f"{insight_data['summary']}\n"

            report += "\n"

        report += "=" * 80 + "\n"
        return report


if __name__ == "__main__":
    # Example usage
    print("Insights module loaded successfully")

    # Demonstrate available insights
    manager = InsightManager()
    print(f"Available insights: {manager.get_available_insights()}")