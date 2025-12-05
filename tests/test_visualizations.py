"""
Unit Tests for Visualizations Module

Comprehensive tests for all visualization strategies and the visualization manager.

Author: Craig
Date: December 2024
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import os

from visualizations import (
    VisualizationStrategy, TimeSeriesPlot, DistributionPlot,
    CategoryPlot, ScatterPlot, CorrelationHeatmap,
    VisualizationManager, save_visualization
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def time_series_data():
    """Create sample time series data."""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'date': dates,
        'sales': np.random.randint(100, 1000, 100),
        'revenue': np.random.uniform(1000, 5000, 100)
    })


@pytest.fixture
def numerical_data():
    """Create sample numerical data."""
    np.random.seed(42)
    return pd.DataFrame({
        'values': np.random.normal(100, 15, 1000),
        'scores': np.random.uniform(0, 100, 1000)
    })


@pytest.fixture
def categorical_data():
    """Create sample categorical data."""
    return pd.DataFrame({
        'category': ['A', 'B', 'C', 'D', 'E'] * 20,
        'values': np.random.randint(10, 100, 100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100)
    })


@pytest.fixture
def scatter_data():
    """Create sample scatter plot data."""
    np.random.seed(42)
    x = np.random.uniform(0, 100, 200)
    y = 2 * x + np.random.normal(0, 10, 200)
    return pd.DataFrame({
        'x_val': x,
        'y_val': y,
        'category': np.random.choice(['A', 'B', 'C'], 200),
        'size': np.random.uniform(10, 100, 200)
    })


@pytest.fixture
def correlation_data():
    """Create sample data for correlation."""
    np.random.seed(42)
    return pd.DataFrame({
        'var1': np.random.normal(50, 10, 100),
        'var2': np.random.normal(100, 20, 100),
        'var3': np.random.normal(75, 15, 100),
        'var4': np.random.normal(60, 12, 100)
    })


# ============================================================================
# TIME SERIES PLOT TESTS
# ============================================================================

class TestTimeSeriesPlot:
    """Test suite for TimeSeriesPlot class."""

    def test_initialization(self):
        """Test TimeSeriesPlot initialization."""
        plot = TimeSeriesPlot()
        assert plot is not None

    def test_get_required_params(self):
        """Test getting required parameters."""
        plot = TimeSeriesPlot()
        params = plot.get_required_params()
        assert 'date_column' in params
        assert 'value_column' in params

    def test_create_matplotlib_basic(self, time_series_data):
        """Test creating basic matplotlib time series plot."""
        plot = TimeSeriesPlot()
        fig = plot.create(time_series_data,
                          date_column='date',
                          value_column='sales',
                          backend='matplotlib')

        assert fig is not None
        assert hasattr(fig, 'savefig')
        plt.close(fig)

    def test_create_plotly_basic(self, time_series_data):
        """Test creating basic plotly time series plot."""
        plot = TimeSeriesPlot()
        fig = plot.create(time_series_data,
                          date_column='date',
                          value_column='sales',
                          backend='plotly')

        assert fig is not None
        assert hasattr(fig, 'write_html')

    def test_aggregation_sum(self, time_series_data):
        """Test time series with sum aggregation."""
        plot = TimeSeriesPlot()
        fig = plot.create(time_series_data,
                          date_column='date',
                          value_column='sales',
                          aggregation='sum',
                          backend='matplotlib')

        assert fig is not None
        plt.close(fig)

    def test_aggregation_mean(self, time_series_data):
        """Test time series with mean aggregation."""
        plot = TimeSeriesPlot()
        fig = plot.create(time_series_data,
                          date_column='date',
                          value_column='sales',
                          aggregation='mean',
                          backend='matplotlib')

        assert fig is not None
        plt.close(fig)

    def test_invalid_date_column(self, time_series_data):
        """Test with invalid date column."""
        plot = TimeSeriesPlot()
        with pytest.raises(ValueError):
            plot.create(time_series_data,
                        date_column='nonexistent',
                        value_column='sales')

    def test_invalid_backend(self, time_series_data):
        """Test with invalid backend."""
        plot = TimeSeriesPlot()
        with pytest.raises(ValueError, match="Unsupported backend"):
            plot.create(time_series_data,
                        date_column='date',
                        value_column='sales',
                        backend='invalid')


# ============================================================================
# DISTRIBUTION PLOT TESTS
# ============================================================================

class TestDistributionPlot:
    """Test suite for DistributionPlot class."""

    def test_initialization(self):
        """Test DistributionPlot initialization."""
        plot = DistributionPlot()
        assert plot is not None

    def test_get_required_params(self):
        """Test getting required parameters."""
        plot = DistributionPlot()
        params = plot.get_required_params()
        assert 'column' in params

    def test_create_histogram_matplotlib(self, numerical_data):
        """Test creating histogram with matplotlib."""
        plot = DistributionPlot()
        fig = plot.create(numerical_data,
                          column='values',
                          plot_type='histogram',
                          backend='matplotlib')

        assert fig is not None
        plt.close(fig)

    def test_create_box_matplotlib(self, numerical_data):
        """Test creating box plot with matplotlib."""
        plot = DistributionPlot()
        fig = plot.create(numerical_data,
                          column='values',
                          plot_type='box',
                          backend='matplotlib')

        assert fig is not None
        plt.close(fig)

    def test_create_violin_matplotlib(self, numerical_data):
        """Test creating violin plot with matplotlib."""
        plot = DistributionPlot()
        fig = plot.create(numerical_data,
                          column='values',
                          plot_type='violin',
                          backend='matplotlib')

        assert fig is not None
        plt.close(fig)

    def test_create_histogram_plotly(self, numerical_data):
        """Test creating histogram with plotly."""
        plot = DistributionPlot()
        fig = plot.create(numerical_data,
                          column='values',
                          plot_type='histogram',
                          backend='plotly')

        assert fig is not None

    def test_custom_bins(self, numerical_data):
        """Test histogram with custom bins."""
        plot = DistributionPlot()
        fig = plot.create(numerical_data,
                          column='values',
                          plot_type='histogram',
                          bins=50,
                          backend='matplotlib')

        assert fig is not None
        plt.close(fig)

    def test_invalid_column(self, numerical_data):
        """Test with invalid column."""
        plot = DistributionPlot()
        with pytest.raises(ValueError):
            plot.create(numerical_data, column='nonexistent')

    def test_invalid_plot_type(self, numerical_data):
        """Test with invalid plot type."""
        plot = DistributionPlot()
        with pytest.raises(ValueError, match="Unsupported plot type"):
            plot.create(numerical_data,
                        column='values',
                        plot_type='invalid',
                        backend='matplotlib')


# ============================================================================
# CATEGORY PLOT TESTS
# ============================================================================

class TestCategoryPlot:
    """Test suite for CategoryPlot class."""

    def test_initialization(self):
        """Test CategoryPlot initialization."""
        plot = CategoryPlot()
        assert plot is not None

    def test_get_required_params(self):
        """Test getting required parameters."""
        plot = CategoryPlot()
        params = plot.get_required_params()
        assert 'column' in params

    def test_create_bar_matplotlib(self, categorical_data):
        """Test creating bar chart with matplotlib."""
        plot = CategoryPlot()
        fig = plot.create(categorical_data,
                          column='category',
                          plot_type='bar',
                          backend='matplotlib')

        assert fig is not None
        plt.close(fig)

    def test_create_pie_matplotlib(self, categorical_data):
        """Test creating pie chart with matplotlib."""
        plot = CategoryPlot()
        fig = plot.create(categorical_data,
                          column='category',
                          plot_type='pie',
                          backend='matplotlib')

        assert fig is not None
        plt.close(fig)

    def test_create_bar_plotly(self, categorical_data):
        """Test creating bar chart with plotly."""
        plot = CategoryPlot()
        fig = plot.create(categorical_data,
                          column='category',
                          plot_type='bar',
                          backend='plotly')

        assert fig is not None

    def test_aggregation_sum(self, categorical_data):
        """Test with sum aggregation."""
        plot = CategoryPlot()
        fig = plot.create(categorical_data,
                          column='category',
                          value_column='values',
                          aggregation='sum',
                          backend='matplotlib')

        assert fig is not None
        plt.close(fig)

    def test_top_n_categories(self, categorical_data):
        """Test showing only top N categories."""
        plot = CategoryPlot()
        fig = plot.create(categorical_data,
                          column='category',
                          top_n=3,
                          backend='matplotlib')

        assert fig is not None
        plt.close(fig)

    def test_invalid_plot_type(self, categorical_data):
        """Test with invalid plot type."""
        plot = CategoryPlot()
        with pytest.raises(ValueError, match="Unsupported plot type"):
            plot.create(categorical_data,
                        column='category',
                        plot_type='invalid',
                        backend='matplotlib')


# ============================================================================
# SCATTER PLOT TESTS
# ============================================================================

class TestScatterPlot:
    """Test suite for ScatterPlot class."""

    def test_initialization(self):
        """Test ScatterPlot initialization."""
        plot = ScatterPlot()
        assert plot is not None

    def test_get_required_params(self):
        """Test getting required parameters."""
        plot = ScatterPlot()
        params = plot.get_required_params()
        assert 'x_column' in params
        assert 'y_column' in params

    def test_create_basic_matplotlib(self, scatter_data):
        """Test creating basic scatter plot with matplotlib."""
        plot = ScatterPlot()
        fig = plot.create(scatter_data,
                          x_column='x_val',
                          y_column='y_val',
                          backend='matplotlib')

        assert fig is not None
        plt.close(fig)

    def test_create_basic_plotly(self, scatter_data):
        """Test creating basic scatter plot with plotly."""
        plot = ScatterPlot()
        fig = plot.create(scatter_data,
                          x_column='x_val',
                          y_column='y_val',
                          backend='plotly')

        assert fig is not None

    def test_with_color_column(self, scatter_data):
        """Test scatter plot with color coding."""
        plot = ScatterPlot()
        fig = plot.create(scatter_data,
                          x_column='x_val',
                          y_column='y_val',
                          color_column='category',
                          backend='matplotlib')

        assert fig is not None
        plt.close(fig)

    def test_with_size_column(self, scatter_data):
        """Test scatter plot with size coding."""
        plot = ScatterPlot()
        fig = plot.create(scatter_data,
                          x_column='x_val',
                          y_column='y_val',
                          size_column='size',
                          backend='matplotlib')

        assert fig is not None
        plt.close(fig)

    def test_with_trend_line(self, scatter_data):
        """Test scatter plot with trend line."""
        plot = ScatterPlot()
        fig = plot.create(scatter_data,
                          x_column='x_val',
                          y_column='y_val',
                          show_trend=True,
                          backend='matplotlib')

        assert fig is not None
        plt.close(fig)

    def test_invalid_columns(self, scatter_data):
        """Test with invalid columns."""
        plot = ScatterPlot()
        with pytest.raises(ValueError):
            plot.create(scatter_data,
                        x_column='nonexistent',
                        y_column='y_val')


# ============================================================================
# CORRELATION HEATMAP TESTS
# ============================================================================

class TestCorrelationHeatmap:
    """Test suite for CorrelationHeatmap class."""

    def test_initialization(self):
        """Test CorrelationHeatmap initialization."""
        plot = CorrelationHeatmap()
        assert plot is not None

    def test_get_required_params(self):
        """Test getting required parameters."""
        plot = CorrelationHeatmap()
        params = plot.get_required_params()
        assert isinstance(params, list)

    def test_create_matplotlib(self, correlation_data):
        """Test creating correlation heatmap with matplotlib."""
        plot = CorrelationHeatmap()
        fig = plot.create(correlation_data, backend='matplotlib')

        assert fig is not None
        plt.close(fig)

    def test_create_plotly(self, correlation_data):
        """Test creating correlation heatmap with plotly."""
        plot = CorrelationHeatmap()
        fig = plot.create(correlation_data, backend='plotly')

        assert fig is not None

    def test_with_specific_columns(self, correlation_data):
        """Test heatmap with specific columns."""
        plot = CorrelationHeatmap()
        fig = plot.create(correlation_data,
                          columns=['var1', 'var2', 'var3'],
                          backend='matplotlib')

        assert fig is not None
        plt.close(fig)

    def test_spearman_correlation(self, correlation_data):
        """Test with Spearman correlation."""
        plot = CorrelationHeatmap()
        fig = plot.create(correlation_data,
                          method='spearman',
                          backend='matplotlib')

        assert fig is not None
        plt.close(fig)

    def test_insufficient_columns(self):
        """Test with insufficient numerical columns."""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        plot = CorrelationHeatmap()

        with pytest.raises(ValueError, match="at least 2 numerical columns"):
            plot.create(df)


# ============================================================================
# VISUALIZATION MANAGER TESTS
# ============================================================================

class TestVisualizationManager:
    """Test suite for VisualizationManager class."""

    def test_initialization(self):
        """Test VisualizationManager initialization."""
        manager = VisualizationManager()
        assert manager is not None
        assert len(manager.strategies) >= 5

    def test_get_available_visualizations(self):
        """Test getting available visualizations."""
        manager = VisualizationManager()
        available = manager.get_available_visualizations()

        assert 'time_series' in available
        assert 'distribution' in available
        assert 'category' in available
        assert 'scatter' in available
        assert 'correlation' in available

    def test_create_time_series(self, time_series_data):
        """Test creating time series through manager."""
        manager = VisualizationManager()
        fig = manager.create_visualization(
            'time_series',
            time_series_data,
            date_column='date',
            value_column='sales',
            backend='matplotlib'
        )

        assert fig is not None
        plt.close(fig)

    def test_create_distribution(self, numerical_data):
        """Test creating distribution through manager."""
        manager = VisualizationManager()
        fig = manager.create_visualization(
            'distribution',
            numerical_data,
            column='values',
            backend='matplotlib'
        )

        assert fig is not None
        plt.close(fig)

    def test_create_category(self, categorical_data):
        """Test creating category plot through manager."""
        manager = VisualizationManager()
        fig = manager.create_visualization(
            'category',
            categorical_data,
            column='category',
            backend='matplotlib'
        )

        assert fig is not None
        plt.close(fig)

    def test_create_scatter(self, scatter_data):
        """Test creating scatter plot through manager."""
        manager = VisualizationManager()
        fig = manager.create_visualization(
            'scatter',
            scatter_data,
            x_column='x_val',
            y_column='y_val',
            backend='matplotlib'
        )

        assert fig is not None
        plt.close(fig)

    def test_create_correlation(self, correlation_data):
        """Test creating correlation heatmap through manager."""
        manager = VisualizationManager()
        fig = manager.create_visualization(
            'correlation',
            correlation_data,
            backend='matplotlib'
        )

        assert fig is not None
        plt.close(fig)

    def test_unsupported_visualization_type(self, numerical_data):
        """Test with unsupported visualization type."""
        manager = VisualizationManager()

        with pytest.raises(ValueError, match="Unsupported visualization type"):
            manager.create_visualization('invalid_type', numerical_data)

    def test_add_strategy(self):
        """Test adding new strategy."""
        manager = VisualizationManager()
        initial_count = len(manager.strategies)

        # Create mock strategy
        class MockStrategy(VisualizationStrategy):
            def create(self, df, **kwargs):
                return None

            def get_required_params(self):
                return []

        manager.add_strategy('mock', MockStrategy())
        assert len(manager.strategies) == initial_count + 1
        assert 'mock' in manager.get_available_visualizations()

    def test_get_required_params(self):
        """Test getting required params for visualization type."""
        manager = VisualizationManager()
        params = manager.get_required_params('time_series')

        assert isinstance(params, list)
        assert 'date_column' in params
        assert 'value_column' in params

    def test_get_required_params_invalid_type(self):
        """Test getting params for invalid type."""
        manager = VisualizationManager()

        with pytest.raises(ValueError):
            manager.get_required_params('invalid_type')


# ============================================================================
# SAVE VISUALIZATION TESTS
# ============================================================================

class TestSaveVisualization:
    """Test suite for save_visualization function."""

    def test_save_matplotlib_png(self, numerical_data):
        """Test saving matplotlib figure as PNG."""
        plot = DistributionPlot()
        fig = plot.create(numerical_data, column='values', backend='matplotlib')

        temp_path = tempfile.mktemp(suffix='.png')

        try:
            result = save_visualization(fig, temp_path, format='png')
            assert result is True
            assert os.path.exists(temp_path)
        finally:
            plt.close(fig)
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_save_matplotlib_pdf(self, numerical_data):
        """Test saving matplotlib figure as PDF."""
        plot = DistributionPlot()
        fig = plot.create(numerical_data, column='values', backend='matplotlib')

        temp_path = tempfile.mktemp(suffix='.pdf')

        try:
            result = save_visualization(fig, temp_path, format='pdf')
            assert result is True
            assert os.path.exists(temp_path)
        finally:
            plt.close(fig)
            if os.path.exists(temp_path):
                os.remove(temp_path)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])