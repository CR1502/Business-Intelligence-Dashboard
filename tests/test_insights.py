"""
Unit Tests for Insights Module

Comprehensive tests for all insight strategies and the insight manager.

Author: Craig
Date: December 2024
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from insights import (
    InsightStrategy, TopBottomPerformers, TrendAnalysis,
    AnomalyDetection, DistributionInsights, CorrelationInsights,
    InsightManager
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sales_data():
    """Create sample sales data."""
    return pd.DataFrame({
        'product': ['A', 'B', 'C', 'D', 'E'] * 20,
        'sales': np.random.randint(100, 1000, 100),
        'revenue': np.random.uniform(1000, 5000, 100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
    })


@pytest.fixture
def time_series_data():
    """Create sample time series data."""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    values = np.cumsum(np.random.randn(100)) + 100  # Random walk with trend
    return pd.DataFrame({
        'date': dates,
        'value': values,
        'sales': np.random.randint(50, 200, 100)
    })


@pytest.fixture
def anomaly_data():
    """Create data with anomalies."""
    # Normal data with a few outliers
    normal = np.random.normal(100, 10, 95)
    outliers = np.array([200, 10, 250, 5, 220])
    data = np.concatenate([normal, outliers])
    np.random.shuffle(data)

    return pd.DataFrame({
        'values': data,
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })


@pytest.fixture
def correlation_data():
    """Create data with correlations."""
    np.random.seed(42)
    x = np.random.normal(50, 10, 100)
    y = 2 * x + np.random.normal(0, 5, 100)  # Strong positive correlation
    z = -1.5 * x + np.random.normal(0, 8, 100)  # Strong negative correlation
    w = np.random.normal(100, 15, 100)  # No correlation

    return pd.DataFrame({
        'var_x': x,
        'var_y': y,
        'var_z': z,
        'var_w': w
    })


@pytest.fixture
def mixed_data():
    """Create data with mixed types."""
    return pd.DataFrame({
        'numerical': np.random.normal(100, 15, 100),
        'categorical': np.random.choice(['Cat1', 'Cat2', 'Cat3'], 100),
        'date': pd.date_range('2024-01-01', periods=100),
        'sales': np.random.randint(50, 500, 100)
    })


# ============================================================================
# TOP/BOTTOM PERFORMERS TESTS
# ============================================================================

class TestTopBottomPerformers:
    """Test suite for TopBottomPerformers class."""

    def test_initialization(self):
        """Test TopBottomPerformers initialization."""
        insight = TopBottomPerformers()
        assert insight is not None

    def test_get_insight_type(self):
        """Test getting insight type."""
        insight = TopBottomPerformers()
        assert insight.get_insight_type() == "top_bottom_performers"

    def test_generate_simple(self, sales_data):
        """Test generating simple top/bottom insights."""
        insight = TopBottomPerformers()
        result = insight.generate(sales_data, column='sales')

        assert result['type'] == 'top_bottom_performers'
        assert 'top_performers' in result
        assert 'bottom_performers' in result
        assert 'summary' in result

    def test_generate_with_groupby(self, sales_data):
        """Test generating insights with groupby."""
        insight = TopBottomPerformers()
        result = insight.generate(
            sales_data,
            column='sales',
            group_by='product',
            aggregation='sum'
        )

        assert result['group_by'] == 'product'
        assert result['aggregation'] == 'sum'
        assert len(result['top_performers']['data']) > 0

    def test_generate_with_custom_n(self, sales_data):
        """Test with custom top_n and bottom_n."""
        insight = TopBottomPerformers()
        result = insight.generate(
            sales_data,
            column='sales',
            top_n=3,
            bottom_n=3
        )

        assert result['top_performers']['count'] <= 3
        assert result['bottom_performers']['count'] <= 3

    def test_invalid_column(self, sales_data):
        """Test with invalid column."""
        insight = TopBottomPerformers()
        with pytest.raises(ValueError):
            insight.generate(sales_data, column='nonexistent')


# ============================================================================
# TREND ANALYSIS TESTS
# ============================================================================

class TestTrendAnalysis:
    """Test suite for TrendAnalysis class."""

    def test_initialization(self):
        """Test TrendAnalysis initialization."""
        insight = TrendAnalysis()
        assert insight is not None

    def test_get_insight_type(self):
        """Test getting insight type."""
        insight = TrendAnalysis()
        assert insight.get_insight_type() == "trend_analysis"

    def test_generate_trend(self, time_series_data):
        """Test generating trend insights."""
        insight = TrendAnalysis()
        result = insight.generate(
            time_series_data,
            date_column='date',
            value_column='value'
        )

        assert result['type'] == 'trend_analysis'
        assert 'trend_direction' in result
        assert 'metrics' in result
        assert 'date_range' in result
        assert 'summary' in result

    def test_trend_metrics(self, time_series_data):
        """Test trend metrics calculation."""
        insight = TrendAnalysis()
        result = insight.generate(
            time_series_data,
            date_column='date',
            value_column='value'
        )

        metrics = result['metrics']
        assert 'first_value' in metrics
        assert 'last_value' in metrics
        assert 'absolute_change' in metrics
        assert 'percentage_change' in metrics
        assert 'growth_rate' in metrics
        assert 'volatility' in metrics

    def test_insufficient_data(self):
        """Test with insufficient data."""
        df = pd.DataFrame({
            'date': [pd.Timestamp('2024-01-01')],
            'value': [100]
        })

        insight = TrendAnalysis()
        result = insight.generate(df, date_column='date', value_column='value')

        assert 'error' in result

    def test_invalid_columns(self, time_series_data):
        """Test with invalid columns."""
        insight = TrendAnalysis()
        with pytest.raises(ValueError):
            insight.generate(
                time_series_data,
                date_column='nonexistent',
                value_column='value'
            )


# ============================================================================
# ANOMALY DETECTION TESTS
# ============================================================================

class TestAnomalyDetection:
    """Test suite for AnomalyDetection class."""

    def test_initialization(self):
        """Test AnomalyDetection initialization."""
        insight = AnomalyDetection()
        assert insight is not None

    def test_get_insight_type(self):
        """Test getting insight type."""
        insight = AnomalyDetection()
        assert insight.get_insight_type() == "anomaly_detection"

    def test_detect_zscore(self, anomaly_data):
        """Test Z-score anomaly detection."""
        insight = AnomalyDetection()
        result = insight.generate(
            anomaly_data,
            column='values',
            method='zscore',
            threshold=2.5
        )

        assert result['type'] == 'anomaly_detection'
        assert result['method'] == 'zscore'
        assert 'statistics' in result
        assert 'anomalies' in result

    def test_detect_iqr(self, anomaly_data):
        """Test IQR anomaly detection."""
        insight = AnomalyDetection()
        result = insight.generate(
            anomaly_data,
            column='values',
            method='iqr',
            threshold=1.5
        )

        assert result['method'] == 'iqr'
        assert result['statistics']['anomaly_count'] >= 0

    def test_no_anomalies(self):
        """Test when no anomalies are found."""
        df = pd.DataFrame({
            'values': np.random.normal(100, 1, 100)  # Very tight distribution
        })

        insight = AnomalyDetection()
        result = insight.generate(df, column='values', threshold=10)

        assert result['statistics']['anomaly_count'] == 0

    def test_non_numerical_column(self, sales_data):
        """Test with non-numerical column."""
        insight = AnomalyDetection()
        result = insight.generate(sales_data, column='product')

        assert 'error' in result

    def test_invalid_method(self, anomaly_data):
        """Test with invalid method."""
        insight = AnomalyDetection()
        with pytest.raises(ValueError):
            insight.generate(anomaly_data, column='values', method='invalid')


# ============================================================================
# DISTRIBUTION INSIGHTS TESTS
# ============================================================================

class TestDistributionInsights:
    """Test suite for DistributionInsights class."""

    def test_initialization(self):
        """Test DistributionInsights initialization."""
        insight = DistributionInsights()
        assert insight is not None

    def test_get_insight_type(self):
        """Test getting insight type."""
        insight = DistributionInsights()
        assert insight.get_insight_type() == "distribution_insights"

    def test_numerical_distribution(self, sales_data):
        """Test numerical distribution analysis."""
        insight = DistributionInsights()
        result = insight.generate(sales_data, column='sales')

        assert result['type'] == 'distribution_insights'
        assert result['data_type'] == 'numerical'
        assert 'statistics' in result
        assert 'distribution_shape' in result

    def test_numerical_statistics(self, sales_data):
        """Test numerical statistics calculation."""
        insight = DistributionInsights()
        result = insight.generate(sales_data, column='sales')

        stats = result['statistics']
        assert 'mean' in stats
        assert 'median' in stats
        assert 'std' in stats
        assert 'skewness' in stats
        assert 'kurtosis' in stats

    def test_categorical_distribution(self, sales_data):
        """Test categorical distribution analysis."""
        insight = DistributionInsights()
        result = insight.generate(sales_data, column='product')

        assert result['data_type'] == 'categorical'
        assert 'value_counts' in result
        assert 'most_common' in result['statistics']

    def test_empty_column(self):
        """Test with empty column."""
        df = pd.DataFrame({'col': [np.nan, np.nan, np.nan]})

        insight = DistributionInsights()
        result = insight.generate(df, column='col')

        assert 'error' in result


# ============================================================================
# CORRELATION INSIGHTS TESTS
# ============================================================================

class TestCorrelationInsights:
    """Test suite for CorrelationInsights class."""

    def test_initialization(self):
        """Test CorrelationInsights initialization."""
        insight = CorrelationInsights()
        assert insight is not None

    def test_get_insight_type(self):
        """Test getting insight type."""
        insight = CorrelationInsights()
        assert insight.get_insight_type() == "correlation_insights"

    def test_generate_correlations(self, correlation_data):
        """Test generating correlation insights."""
        insight = CorrelationInsights()
        result = insight.generate(correlation_data, threshold=0.5)

        assert result['type'] == 'correlation_insights'
        assert 'strong_correlations_found' in result
        assert 'correlations' in result

    def test_strong_correlations_found(self, correlation_data):
        """Test that strong correlations are found."""
        insight = CorrelationInsights()
        result = insight.generate(correlation_data, threshold=0.7)

        # Should find strong correlations in our test data
        assert result['strong_correlations_found'] > 0

    def test_correlation_details(self, correlation_data):
        """Test correlation details."""
        insight = CorrelationInsights()
        result = insight.generate(correlation_data, threshold=0.5)

        if len(result['correlations']) > 0:
            corr = result['correlations'][0]
            assert 'variable1' in corr
            assert 'variable2' in corr
            assert 'correlation' in corr
            assert 'strength' in corr
            assert 'direction' in corr

    def test_different_methods(self, correlation_data):
        """Test different correlation methods."""
        insight = CorrelationInsights()

        # Pearson
        result1 = insight.generate(correlation_data, method='pearson')
        assert result1['method'] == 'pearson'

        # Spearman
        result2 = insight.generate(correlation_data, method='spearman')
        assert result2['method'] == 'spearman'

    def test_insufficient_columns(self):
        """Test with insufficient numerical columns."""
        df = pd.DataFrame({'col': [1, 2, 3]})

        insight = CorrelationInsights()
        result = insight.generate(df)

        assert 'error' in result


# ============================================================================
# INSIGHT MANAGER TESTS
# ============================================================================

class TestInsightManager:
    """Test suite for InsightManager class."""

    def test_initialization(self):
        """Test InsightManager initialization."""
        manager = InsightManager()
        assert manager is not None
        assert len(manager.strategies) >= 5

    def test_get_available_insights(self):
        """Test getting available insights."""
        manager = InsightManager()
        available = manager.get_available_insights()

        assert 'top_bottom' in available
        assert 'trend' in available
        assert 'anomaly' in available
        assert 'distribution' in available
        assert 'correlation' in available

    def test_generate_top_bottom(self, sales_data):
        """Test generating top/bottom insight through manager."""
        manager = InsightManager()
        result = manager.generate_insight(
            'top_bottom',
            sales_data,
            column='sales'
        )

        assert result['type'] == 'top_bottom_performers'

    def test_generate_trend(self, time_series_data):
        """Test generating trend insight through manager."""
        manager = InsightManager()
        result = manager.generate_insight(
            'trend',
            time_series_data,
            date_column='date',
            value_column='value'
        )

        assert result['type'] == 'trend_analysis'

    def test_generate_anomaly(self, anomaly_data):
        """Test generating anomaly insight through manager."""
        manager = InsightManager()
        result = manager.generate_insight(
            'anomaly',
            anomaly_data,
            column='values'
        )

        assert result['type'] == 'anomaly_detection'

    def test_generate_distribution(self, sales_data):
        """Test generating distribution insight through manager."""
        manager = InsightManager()
        result = manager.generate_insight(
            'distribution',
            sales_data,
            column='sales'
        )

        assert result['type'] == 'distribution_insights'

    def test_generate_correlation(self, correlation_data):
        """Test generating correlation insight through manager."""
        manager = InsightManager()
        result = manager.generate_insight(
            'correlation',
            correlation_data
        )

        assert result['type'] == 'correlation_insights'

    def test_unsupported_insight_type(self, sales_data):
        """Test with unsupported insight type."""
        manager = InsightManager()

        with pytest.raises(ValueError, match="Unsupported insight type"):
            manager.generate_insight('invalid_type', sales_data)

    def test_generate_all_insights(self, mixed_data):
        """Test generating all insights."""
        manager = InsightManager()
        results = manager.generate_all_insights(mixed_data)

        assert isinstance(results, dict)
        # Should generate at least some insights
        assert len(results) > 0

    def test_add_strategy(self):
        """Test adding new strategy."""
        manager = InsightManager()
        initial_count = len(manager.strategies)

        # Create mock strategy
        class MockStrategy(InsightStrategy):
            def generate(self, df, **kwargs):
                return {'type': 'mock'}

            def get_insight_type(self):
                return 'mock'

        manager.add_strategy('mock', MockStrategy())
        assert len(manager.strategies) == initial_count + 1
        assert 'mock' in manager.get_available_insights()

    def test_format_insight_report(self, sales_data):
        """Test formatting insight report."""
        manager = InsightManager()
        insights = {
            'top_bottom': manager.generate_insight(
                'top_bottom', sales_data, column='sales'
            )
        }

        report = manager.format_insight_report(insights)
        assert isinstance(report, str)
        assert 'INSIGHTS REPORT' in report
        assert 'TOP BOTTOM' in report


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])