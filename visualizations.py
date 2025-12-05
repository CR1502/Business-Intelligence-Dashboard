"""
Visualizations Module for Business Intelligence Dashboard

This module handles all data visualization operations using Strategy Pattern.
Supports multiple chart types with flexible rendering backends.

Author: Craig
Date: December 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Union, List, Dict, Optional, Any, Tuple
from abc import ABC, abstractmethod
import logging
from pathlib import Path

from utils import ColumnValidator, DataFrameValidator, format_number, Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# STRATEGY PATTERN - Visualization Strategies
# Follows Open/Closed Principle and Strategy Pattern
# ============================================================================

class VisualizationStrategy(ABC):
    """
    Abstract base class for visualization strategies.
    Follows Strategy Pattern - allows different visualization algorithms.
    """

    @abstractmethod
    def create(self, df: pd.DataFrame, **kwargs) -> Any:
        """
        Create visualization.

        Args:
            df: DataFrame to visualize
            **kwargs: Additional parameters for visualization

        Returns:
            Visualization object (matplotlib Figure or plotly Figure)
        """
        pass

    @abstractmethod
    def get_required_params(self) -> List[str]:
        """
        Get list of required parameters for this visualization.

        Returns:
            List of required parameter names
        """
        pass


# ============================================================================
# TIME SERIES VISUALIZATIONS
# ============================================================================

class TimeSeriesPlot(VisualizationStrategy):
    """
    Create time series line plots.
    Follows Single Responsibility Principle - only handles time series plots.
    """

    def get_required_params(self) -> List[str]:
        """Required parameters for time series plot."""
        return ['date_column', 'value_column']

    def create(self, df: pd.DataFrame, date_column: str, value_column: str,
               title: str = "Time Series Plot",
               aggregation: str = 'sum',
               backend: str = 'matplotlib',
               **kwargs) -> Any:
        """
        Create time series plot.

        Args:
            df: DataFrame with time series data
            date_column: Column containing dates
            value_column: Column containing values to plot
            title: Plot title
            aggregation: Aggregation method ('sum', 'mean', 'count', 'median')
            backend: Visualization backend ('matplotlib' or 'plotly')
            **kwargs: Additional plotting parameters

        Returns:
            matplotlib Figure or plotly Figure
        """
        # Validate inputs
        DataFrameValidator().validate(df)
        ColumnValidator().validate(df, [date_column, value_column])

        # Prepare data
        df_plot = df.copy()

        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_plot[date_column]):
            df_plot[date_column] = pd.to_datetime(df_plot[date_column], errors='coerce')

        # Remove rows with NaT dates
        df_plot = df_plot.dropna(subset=[date_column])

        # Sort by date
        df_plot = df_plot.sort_values(date_column)

        # Apply aggregation if needed
        if aggregation != 'none':
            df_plot = self._apply_aggregation(df_plot, date_column, value_column, aggregation)

        # Create visualization based on backend
        if backend == 'matplotlib':
            return self._create_matplotlib(df_plot, date_column, value_column, title, aggregation)
        elif backend == 'plotly':
            return self._create_plotly(df_plot, date_column, value_column, title, aggregation)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _apply_aggregation(self, df: pd.DataFrame, date_column: str,
                          value_column: str, aggregation: str) -> pd.DataFrame:
        """Apply aggregation to time series data."""
        if aggregation == 'sum':
            return df.groupby(date_column)[value_column].sum().reset_index()
        elif aggregation == 'mean':
            return df.groupby(date_column)[value_column].mean().reset_index()
        elif aggregation == 'count':
            return df.groupby(date_column)[value_column].count().reset_index()
        elif aggregation == 'median':
            return df.groupby(date_column)[value_column].median().reset_index()
        else:
            return df

    def _create_matplotlib(self, df: pd.DataFrame, date_column: str,
                          value_column: str, title: str, aggregation: str):
        """Create matplotlib time series plot."""
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(df[date_column], df[value_column], marker='o', linewidth=2, markersize=4)
        ax.set_xlabel(date_column, fontsize=12)
        ax.set_ylabel(f"{value_column} ({aggregation})", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        logger.info(f"Created matplotlib time series plot: {title}")
        return fig

    def _create_plotly(self, df: pd.DataFrame, date_column: str,
                      value_column: str, title: str, aggregation: str):
        """Create plotly time series plot."""
        fig = px.line(df, x=date_column, y=value_column,
                     title=title,
                     labels={value_column: f"{value_column} ({aggregation})"})

        fig.update_traces(mode='lines+markers')
        fig.update_layout(
            xaxis_title=date_column,
            yaxis_title=f"{value_column} ({aggregation})",
            hovermode='x unified',
            template='plotly_white'
        )

        logger.info(f"Created plotly time series plot: {title}")
        return fig


# ============================================================================
# DISTRIBUTION VISUALIZATIONS
# ============================================================================

class DistributionPlot(VisualizationStrategy):
    """
    Create distribution plots (histogram, box plot, violin plot).
    Follows Single Responsibility Principle - only handles distribution plots.
    """

    def get_required_params(self) -> List[str]:
        """Required parameters for distribution plot."""
        return ['column']

    def create(self, df: pd.DataFrame, column: str,
               plot_type: str = 'histogram',
               title: str = "Distribution Plot",
               bins: int = 30,
               backend: str = 'matplotlib',
               **kwargs) -> Any:
        """
        Create distribution plot.

        Args:
            df: DataFrame with data
            column: Column to visualize
            plot_type: Type of plot ('histogram', 'box', 'violin')
            title: Plot title
            bins: Number of bins for histogram
            backend: Visualization backend ('matplotlib' or 'plotly')
            **kwargs: Additional plotting parameters

        Returns:
            matplotlib Figure or plotly Figure
        """
        # Validate inputs
        DataFrameValidator().validate(df)
        ColumnValidator().validate(df, column)

        # Remove NaN values
        df_plot = df[column].dropna()

        if len(df_plot) == 0:
            raise ValueError(f"No valid data in column '{column}'")

        # Create visualization based on backend
        if backend == 'matplotlib':
            return self._create_matplotlib(df_plot, column, plot_type, title, bins)
        elif backend == 'plotly':
            return self._create_plotly(df_plot, column, plot_type, title, bins)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _create_matplotlib(self, data: pd.Series, column: str,
                          plot_type: str, title: str, bins: int):
        """Create matplotlib distribution plot."""
        fig, ax = plt.subplots(figsize=(10, 6))

        if plot_type == 'histogram':
            ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)
            ax.set_ylabel('Frequency', fontsize=12)

        elif plot_type == 'box':
            ax.boxplot(data, vert=True)
            ax.set_ylabel(column, fontsize=12)

        elif plot_type == 'violin':
            # Use seaborn for violin plot
            sns.violinplot(y=data, ax=ax)
            ax.set_ylabel(column, fontsize=12)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

        ax.set_xlabel(column if plot_type == 'histogram' else '', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        logger.info(f"Created matplotlib {plot_type} plot: {title}")
        return fig

    def _create_plotly(self, data: pd.Series, column: str,
                      plot_type: str, title: str, bins: int):
        """Create plotly distribution plot."""
        if plot_type == 'histogram':
            fig = px.histogram(data, x=data.values, nbins=bins, title=title,
                             labels={'x': column, 'y': 'Frequency'})

        elif plot_type == 'box':
            fig = px.box(y=data.values, title=title, labels={'y': column})

        elif plot_type == 'violin':
            fig = px.violin(y=data.values, title=title, labels={'y': column})
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

        fig.update_layout(template='plotly_white')
        logger.info(f"Created plotly {plot_type} plot: {title}")
        return fig


# ============================================================================
# CATEGORY VISUALIZATIONS
# ============================================================================

class CategoryPlot(VisualizationStrategy):
    """
    Create category plots (bar chart, pie chart).
    Follows Single Responsibility Principle - only handles category plots.
    """

    def get_required_params(self) -> List[str]:
        """Required parameters for category plot."""
        return ['column']

    def create(self, df: pd.DataFrame, column: str,
               value_column: Optional[str] = None,
               plot_type: str = 'bar',
               title: str = "Category Analysis",
               aggregation: str = 'count',
               top_n: Optional[int] = None,
               backend: str = 'matplotlib',
               **kwargs) -> Any:
        """
        Create category plot.

        Args:
            df: DataFrame with data
            column: Categorical column to visualize
            value_column: Optional value column for aggregation
            plot_type: Type of plot ('bar' or 'pie')
            title: Plot title
            aggregation: Aggregation method ('count', 'sum', 'mean', 'median')
            top_n: Show only top N categories
            backend: Visualization backend ('matplotlib' or 'plotly')
            **kwargs: Additional plotting parameters

        Returns:
            matplotlib Figure or plotly Figure
        """
        # Validate inputs
        DataFrameValidator().validate(df)
        ColumnValidator().validate(df, column)

        if value_column:
            ColumnValidator().validate(df, value_column)

        # Prepare data
        if value_column and aggregation != 'count':
            # Aggregate by category
            if aggregation == 'sum':
                data = df.groupby(column)[value_column].sum()
            elif aggregation == 'mean':
                data = df.groupby(column)[value_column].mean()
            elif aggregation == 'median':
                data = df.groupby(column)[value_column].median()
            else:
                data = df[column].value_counts()
        else:
            # Simple count
            data = df[column].value_counts()

        # Get top N if specified
        if top_n:
            data = data.nlargest(top_n)

        # Sort for better visualization
        data = data.sort_values(ascending=False)

        # Create visualization based on backend
        if backend == 'matplotlib':
            return self._create_matplotlib(data, column, plot_type, title, aggregation)
        elif backend == 'plotly':
            return self._create_plotly(data, column, plot_type, title, aggregation)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _create_matplotlib(self, data: pd.Series, column: str,
                          plot_type: str, title: str, aggregation: str):
        """Create matplotlib category plot."""
        fig, ax = plt.subplots(figsize=(10, 6))

        if plot_type == 'bar':
            bars = ax.bar(range(len(data)), data.values, edgecolor='black', alpha=0.7)
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(data.index, rotation=45, ha='right')
            ax.set_xlabel(column, fontsize=12)
            ax.set_ylabel(f'Value ({aggregation})', fontsize=12)

            # Add value labels on bars
            for i, (idx, val) in enumerate(data.items()):
                ax.text(i, val, format_number(val), ha='center', va='bottom')

        elif plot_type == 'pie':
            wedges, texts, autotexts = ax.pie(data.values, labels=data.index,
                                               autopct='%1.1f%%', startangle=90)
            # Make percentage text more readable
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        logger.info(f"Created matplotlib {plot_type} plot: {title}")
        return fig

    def _create_plotly(self, data: pd.Series, column: str,
                      plot_type: str, title: str, aggregation: str):
        """Create plotly category plot."""
        if plot_type == 'bar':
            fig = px.bar(x=data.index, y=data.values, title=title,
                        labels={'x': column, 'y': f'Value ({aggregation})'})
            fig.update_traces(text=data.values, textposition='outside')

        elif plot_type == 'pie':
            fig = px.pie(values=data.values, names=data.index, title=title)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

        fig.update_layout(template='plotly_white')
        logger.info(f"Created plotly {plot_type} plot: {title}")
        return fig


# ============================================================================
# RELATIONSHIP VISUALIZATIONS
# ============================================================================

class ScatterPlot(VisualizationStrategy):
    """
    Create scatter plots to show relationships between variables.
    Follows Single Responsibility Principle - only handles scatter plots.
    """

    def get_required_params(self) -> List[str]:
        """Required parameters for scatter plot."""
        return ['x_column', 'y_column']

    def create(self, df: pd.DataFrame, x_column: str, y_column: str,
               title: str = "Scatter Plot",
               color_column: Optional[str] = None,
               size_column: Optional[str] = None,
               show_trend: bool = False,
               backend: str = 'matplotlib',
               **kwargs) -> Any:
        """
        Create scatter plot.

        Args:
            df: DataFrame with data
            x_column: Column for x-axis
            y_column: Column for y-axis
            title: Plot title
            color_column: Optional column for color coding
            size_column: Optional column for point sizes
            show_trend: Whether to show trend line
            backend: Visualization backend ('matplotlib' or 'plotly')
            **kwargs: Additional plotting parameters

        Returns:
            matplotlib Figure or plotly Figure
        """
        # Validate inputs
        DataFrameValidator().validate(df)
        ColumnValidator().validate(df, [x_column, y_column])

        if color_column:
            ColumnValidator().validate(df, color_column)
        if size_column:
            ColumnValidator().validate(df, size_column)

        # Remove rows with NaN in required columns
        required_cols = [x_column, y_column]
        if color_column:
            required_cols.append(color_column)
        if size_column:
            required_cols.append(size_column)

        df_plot = df[required_cols].dropna()

        if len(df_plot) == 0:
            raise ValueError("No valid data after removing NaN values")

        # Create visualization based on backend
        if backend == 'matplotlib':
            return self._create_matplotlib(df_plot, x_column, y_column, title,
                                          color_column, size_column, show_trend)
        elif backend == 'plotly':
            return self._create_plotly(df_plot, x_column, y_column, title,
                                      color_column, size_column, show_trend)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _create_matplotlib(self, df: pd.DataFrame, x_column: str, y_column: str,
                          title: str, color_column: Optional[str],
                          size_column: Optional[str], show_trend: bool):
        """Create matplotlib scatter plot."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Prepare scatter parameters
        scatter_kwargs = {'alpha': 0.6, 'edgecolors': 'black', 'linewidth': 0.5}

        if size_column:
            scatter_kwargs['s'] = df[size_column]
        else:
            scatter_kwargs['s'] = 50

        if color_column:
            # Check if color column is categorical (string type)
            if df[color_column].dtype == 'object' or pd.api.types.is_categorical_dtype(df[color_column]):
                # Convert categorical to numerical codes for matplotlib
                categories = df[color_column].astype('category')
                color_codes = categories.cat.codes
                scatter = ax.scatter(df[x_column], df[y_column], c=color_codes,
                                   cmap='viridis', **scatter_kwargs)
                # Create custom legend
                handles = []
                for i, cat in enumerate(categories.cat.categories):
                    handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                             markerfacecolor=plt.cm.viridis(i / len(categories.cat.categories)),
                                             markersize=8, label=cat))
                ax.legend(handles=handles, title=color_column)
            else:
                # Numerical color column
                scatter = ax.scatter(df[x_column], df[y_column], c=df[color_column],
                                   cmap='viridis', **scatter_kwargs)
                plt.colorbar(scatter, ax=ax, label=color_column)
        else:
            ax.scatter(df[x_column], df[y_column], **scatter_kwargs)

        # Add trend line if requested
        if show_trend:
            z = np.polyfit(df[x_column], df[y_column], 1)
            p = np.poly1d(z)
            ax.plot(df[x_column], p(df[x_column]), "r--", alpha=0.8, label='Trend')
            ax.legend()

        ax.set_xlabel(x_column, fontsize=12)
        ax.set_ylabel(y_column, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        logger.info(f"Created matplotlib scatter plot: {title}")
        return fig

    def _create_plotly(self, df: pd.DataFrame, x_column: str, y_column: str,
                      title: str, color_column: Optional[str],
                      size_column: Optional[str], show_trend: bool):
        """Create plotly scatter plot."""
        fig = px.scatter(df, x=x_column, y=y_column,
                        color=color_column, size=size_column,
                        title=title,
                        trendline='ols' if show_trend else None)

        fig.update_layout(template='plotly_white')
        logger.info(f"Created plotly scatter plot: {title}")
        return fig


class CorrelationHeatmap(VisualizationStrategy):
    """
    Create correlation heatmap for numerical variables.
    Follows Single Responsibility Principle - only handles correlation heatmaps.
    """

    def get_required_params(self) -> List[str]:
        """Required parameters for correlation heatmap."""
        return []  # Uses all numerical columns by default

    def create(self, df: pd.DataFrame,
               columns: Optional[List[str]] = None,
               title: str = "Correlation Heatmap",
               method: str = 'pearson',
               backend: str = 'matplotlib',
               **kwargs) -> Any:
        """
        Create correlation heatmap.

        Args:
            df: DataFrame with data
            columns: Optional list of columns to include
            title: Plot title
            method: Correlation method ('pearson', 'spearman', 'kendall')
            backend: Visualization backend ('matplotlib' or 'plotly')
            **kwargs: Additional plotting parameters

        Returns:
            matplotlib Figure or plotly Figure
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
            raise ValueError("Need at least 2 numerical columns for correlation heatmap")

        # Calculate correlation
        corr_matrix = df_corr.corr(method=method)

        # Create visualization based on backend
        if backend == 'matplotlib':
            return self._create_matplotlib(corr_matrix, title)
        elif backend == 'plotly':
            return self._create_plotly(corr_matrix, title)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def _create_matplotlib(self, corr_matrix: pd.DataFrame, title: str):
        """Create matplotlib correlation heatmap."""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1,
                   cbar_kws={"shrink": 0.8}, ax=ax)

        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        logger.info(f"Created matplotlib correlation heatmap: {title}")
        return fig

    def _create_plotly(self, corr_matrix: pd.DataFrame, title: str):
        """Create plotly correlation heatmap."""
        fig = px.imshow(corr_matrix,
                       text_auto='.2f',
                       color_continuous_scale='RdBu_r',
                       title=title,
                       aspect='auto')

        fig.update_layout(template='plotly_white')
        logger.info(f"Created plotly correlation heatmap: {title}")
        return fig


# ============================================================================
# VISUALIZATION MANAGER
# Uses Strategy Pattern to manage different visualization types
# ============================================================================

class VisualizationManager:
    """
    Manager class for visualizations using Strategy Pattern.
    Follows Open/Closed Principle - open for extension, closed for modification.
    """

    def __init__(self):
        """Initialize VisualizationManager with all available strategies."""
        self.strategies: Dict[str, VisualizationStrategy] = {
            'time_series': TimeSeriesPlot(),
            'distribution': DistributionPlot(),
            'category': CategoryPlot(),
            'scatter': ScatterPlot(),
            'correlation': CorrelationHeatmap()
        }

    def create_visualization(self, viz_type: str, df: pd.DataFrame, **kwargs) -> Any:
        """
        Create visualization using specified strategy.

        Args:
            viz_type: Type of visualization ('time_series', 'distribution', etc.)
            df: DataFrame to visualize
            **kwargs: Parameters specific to visualization type

        Returns:
            Visualization object

        Raises:
            ValueError: If visualization type is not supported
        """
        if viz_type not in self.strategies:
            raise ValueError(
                f"Unsupported visualization type: {viz_type}. "
                f"Available types: {list(self.strategies.keys())}"
            )

        strategy = self.strategies[viz_type]
        return strategy.create(df, **kwargs)

    def add_strategy(self, name: str, strategy: VisualizationStrategy) -> None:
        """
        Add new visualization strategy.
        Follows Open/Closed Principle - extend functionality without modifying existing code.

        Args:
            name: Name for the strategy
            strategy: Visualization strategy instance
        """
        self.strategies[name] = strategy
        logger.info(f"Added new visualization strategy: {name}")

    def get_available_visualizations(self) -> List[str]:
        """
        Get list of available visualization types.

        Returns:
            List of visualization type names
        """
        return list(self.strategies.keys())

    def get_required_params(self, viz_type: str) -> List[str]:
        """
        Get required parameters for a visualization type.

        Args:
            viz_type: Type of visualization

        Returns:
            List of required parameter names
        """
        if viz_type not in self.strategies:
            raise ValueError(f"Unsupported visualization type: {viz_type}")

        return self.strategies[viz_type].get_required_params()


# ============================================================================
# UTILITY FUNCTIONS FOR SAVING VISUALIZATIONS
# ============================================================================

def save_visualization(fig: Any, filepath: Union[str, Path],
                      dpi: int = 300, format: str = 'png') -> bool:
    """
    Save visualization to file.

    Args:
        fig: Matplotlib or Plotly figure
        filepath: Path to save file
        dpi: DPI for raster formats
        format: File format ('png', 'jpg', 'pdf', 'svg', 'html')

    Returns:
        bool: True if saved successfully
    """
    try:
        filepath = Path(filepath)

        # Handle matplotlib figures
        if hasattr(fig, 'savefig'):
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight', format=format)
            logger.info(f"Saved matplotlib figure to {filepath}")

        # Handle plotly figures
        elif hasattr(fig, 'write_image') or hasattr(fig, 'write_html'):
            if format in ['png', 'jpg', 'pdf', 'svg']:
                fig.write_image(filepath, format=format)
            elif format == 'html':
                fig.write_html(filepath)
            logger.info(f"Saved plotly figure to {filepath}")

        else:
            raise ValueError("Unknown figure type")

        return True

    except Exception as e:
        logger.error(f"Error saving visualization: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    print("Visualizations module loaded successfully")

    # Demonstrate available visualizations
    manager = VisualizationManager()
    print(f"Available visualizations: {manager.get_available_visualizations()}")