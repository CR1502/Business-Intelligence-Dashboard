"""
Business Intelligence Dashboard - Main Gradio Application

This application provides an interactive BI dashboard with automated insights,
visualizations, and data exploration capabilities.

Author: Craig
Date: December 2024
"""

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import logging

from data_processor import DataProcessor, DataProfiler, DataFilter
from visualizations import VisualizationManager, save_visualization
from insights import InsightManager
from utils import (
    get_column_types, format_number, format_percentage,
    Config, CSVExporter, ExcelExporter
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state management
class AppState:
    """
    Manages application state across tabs.
    Follows Single Responsibility Principle - only manages state.
    """

    def __init__(self):
        self.processor = DataProcessor()
        self.viz_manager = VisualizationManager()
        self.insight_manager = InsightManager()

        # Available datasets
        self.datasets = {
            'Online Retail': 'data/Online_Retail.xlsx',
            'Airbnb': 'data/Airbnb.csv'
        }

        # Current session data
        self.current_dataset_name = None
        self.current_df = None
        self.filtered_df = None
        self.active_filters = []
        self.current_recommendations = None

    def load_dataset(self, dataset_name: str, file_path: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
        """
        Load dataset by name or from uploaded file.

        Args:
            dataset_name: Name of dataset to load
            file_path: Optional path to uploaded file

        Returns:
            Tuple of (DataFrame, status_message)
        """
        try:
            if file_path:
                # Load uploaded file
                df = self.processor.load_and_prepare_data(file_path)
                self.current_dataset_name = f"Uploaded: {Path(file_path).name}"
            else:
                # Load predefined dataset
                if dataset_name not in self.datasets:
                    return None, f"❌ Dataset '{dataset_name}' not found"

                file_path = self.datasets[dataset_name]
                df = self.processor.load_and_prepare_data(file_path)
                self.current_dataset_name = dataset_name

            self.current_df = df
            self.filtered_df = df.copy()
            self.active_filters = []
            self.current_recommendations = None

            message = f"✅ Successfully loaded '{self.current_dataset_name}' - {len(df)} rows, {len(df.columns)} columns"
            logger.info(message)
            return df, message

        except Exception as e:
            error_msg = f"❌ Error loading dataset: {str(e)}"
            logger.error(error_msg)
            return None, error_msg

    def get_column_info(self) -> Dict[str, List[str]]:
        """Get categorized column information."""
        if self.current_df is None:
            return {'numerical': [], 'categorical': [], 'datetime': []}
        return get_column_types(self.current_df)

    def apply_filters(self, filters: List[Dict]) -> pd.DataFrame:
        """Apply filters to current dataset."""
        if self.current_df is None:
            return None

        self.active_filters = filters
        self.filtered_df = self.processor.apply_filters(filters)
        return self.filtered_df

    def reset_filters(self) -> pd.DataFrame:
        """Reset all filters."""
        if self.current_df is None:
            return None

        self.filtered_df = self.current_df.copy()
        self.active_filters = []
        return self.filtered_df


# Initialize global state
app_state = AppState()


# ============================================================================
# SMART VISUALIZATION RECOMMENDATIONS
# ============================================================================

class SmartVisualizationRecommender:
    """
    Recommends best visualizations based on data characteristics.
    Follows Single Responsibility Principle - only handles recommendations.
    """

    @staticmethod
    def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze dataset and recommend visualizations.

        Args:
            df: DataFrame to analyze

        Returns:
            Dict with recommendations
        """
        column_types = get_column_types(df)
        recommendations = []

        # Time Series Recommendations
        if len(column_types['datetime']) > 0 and len(column_types['numerical']) > 0:
            recommendations.append({
                'type': 'time_series',
                'priority': 'high',
                'reason': 'Detected date and numerical columns - perfect for trend analysis',
                'suggested_params': {
                    'date_column': column_types['datetime'][0],
                    'value_column': column_types['numerical'][0],
                    'aggregation': 'sum'
                }
            })

        # Correlation Heatmap Recommendations
        if len(column_types['numerical']) >= 3:
            recommendations.append({
                'type': 'correlation',
                'priority': 'high',
                'reason': f'Found {len(column_types["numerical"])} numerical columns - great for correlation analysis',
                'suggested_params': {}
            })

        # Category Analysis Recommendations
        if len(column_types['categorical']) > 0:
            cat_col = column_types['categorical'][0]
            unique_count = df[cat_col].nunique()

            if unique_count <= 10:
                recommendations.append({
                    'type': 'category',
                    'priority': 'high',
                    'reason': f'Found categorical column "{cat_col}" with {unique_count} categories',
                    'suggested_params': {
                        'column': cat_col,
                        'plot_type': 'bar'
                    }
                })

        # Distribution Recommendations
        if len(column_types['numerical']) > 0:
            recommendations.append({
                'type': 'distribution',
                'priority': 'medium',
                'reason': 'Numerical data available - useful for understanding value distribution',
                'suggested_params': {
                    'column': column_types['numerical'][0],
                    'plot_type': 'histogram'
                }
            })

        # Scatter Plot Recommendations
        if len(column_types['numerical']) >= 2:
            recommendations.append({
                'type': 'scatter',
                'priority': 'medium',
                'reason': 'Multiple numerical columns - explore relationships between variables',
                'suggested_params': {
                    'x_column': column_types['numerical'][0],
                    'y_column': column_types['numerical'][1]
                }
            })

        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order[x['priority']])

        return {
            'column_types': column_types,
            'recommendations': recommendations,
            'summary': SmartVisualizationRecommender._generate_summary(recommendations)
        }

    @staticmethod
    def _generate_summary(recommendations: List[Dict]) -> str:
        """Generate human-readable summary of recommendations."""
        if not recommendations:
            return "No specific visualization recommendations available."

        high_priority = [r for r in recommendations if r['priority'] == 'high']

        if high_priority:
            summary = f"🎯 **Top Recommendation**: {high_priority[0]['type'].replace('_', ' ').title()}\n"
            summary += f"💡 {high_priority[0]['reason']}\n\n"

            if len(high_priority) > 1:
                summary += f"Also recommended: {', '.join([r['type'].replace('_', ' ').title() for r in high_priority[1:]])}"
        else:
            summary = f"📊 Recommended: {recommendations[0]['type'].replace('_', ' ').title()}"

        return summary


# ============================================================================
# TAB 1: DATASET SELECTION
# ============================================================================

def create_dataset_tab():
    """Create dataset selection and preview tab."""

    with gr.Tab("📊 Dataset Selection"):
        gr.Markdown("## Select or Upload Dataset")
        gr.Markdown("Choose from pre-loaded datasets or upload your own (CSV, Excel, JSON, Parquet)")

        with gr.Row():
            with gr.Column(scale=1):
                dataset_dropdown = gr.Dropdown(
                    choices=list(app_state.datasets.keys()),
                    label="Pre-loaded Datasets",
                    value=None
                )

                load_btn = gr.Button("📂 Load Selected Dataset", variant="primary")

                gr.Markdown("### OR Upload Your Own Dataset")
                file_upload = gr.File(
                    label="Upload Dataset (Max 50MB)",
                    file_types=[".csv", ".xlsx", ".xls", ".json", ".parquet"]
                )

                upload_btn = gr.Button("📤 Upload & Process", variant="secondary")

            with gr.Column(scale=1):
                status_box = gr.Textbox(
                    label="Status",
                    value="No dataset loaded",
                    interactive=False,
                    lines=3
                )

                dataset_info = gr.Textbox(
                    label="Dataset Information",
                    value="",
                    interactive=False,
                    lines=8
                )

        gr.Markdown("### Data Preview")
        data_preview = gr.Dataframe(
            label="First 100 rows",
            interactive=False,
            wrap=True
        )

        # Event handlers
        def load_predefined_dataset(dataset_name):
            if not dataset_name:
                return None, "⚠️ Please select a dataset", "", None

            df, status = app_state.load_dataset(dataset_name)

            if df is not None:
                info = f"📊 **Dataset**: {dataset_name}\n"
                info += f"📏 **Shape**: {df.shape[0]} rows × {df.shape[1]} columns\n"
                info += f"💾 **Memory**: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB\n\n"
                info += f"**Column Types**:\n"

                col_types = get_column_types(df)
                info += f"- Numerical: {len(col_types['numerical'])}\n"
                info += f"- Categorical: {len(col_types['categorical'])}\n"
                info += f"- DateTime: {len(col_types['datetime'])}\n"

                preview = df.head(100)
                return dataset_name, status, info, preview

            return None, status, "", None

        def upload_custom_dataset(file):
            if file is None:
                return "⚠️ Please upload a file", "", None

            # Check file size (50MB limit)
            file_size_mb = Path(file.name).stat().st_size / (1024 * 1024)
            if file_size_mb > 50:
                return f"❌ File too large ({file_size_mb:.1f}MB). Maximum size: 50MB", "", None

            df, status = app_state.load_dataset("uploaded", file.name)

            if df is not None:
                info = f"📊 **Dataset**: {Path(file.name).name}\n"
                info += f"📏 **Shape**: {df.shape[0]} rows × {df.shape[1]} columns\n"
                info += f"💾 **Memory**: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB\n\n"
                info += f"**Column Types**:\n"

                col_types = get_column_types(df)
                info += f"- Numerical: {len(col_types['numerical'])}\n"
                info += f"- Categorical: {len(col_types['categorical'])}\n"
                info += f"- DateTime: {len(col_types['datetime'])}\n"

                preview = df.head(100)
                return status, info, preview

            return status, "", None

        load_btn.click(
            fn=load_predefined_dataset,
            inputs=[dataset_dropdown],
            outputs=[dataset_dropdown, status_box, dataset_info, data_preview]
        )

        upload_btn.click(
            fn=upload_custom_dataset,
            inputs=[file_upload],
            outputs=[status_box, dataset_info, data_preview]
        )

    return dataset_dropdown, status_box, dataset_info, data_preview, load_btn, upload_btn


# ============================================================================
# TAB 2: STATISTICS & PROFILING
# ============================================================================

def create_statistics_tab():
    """Create statistics and data profiling tab."""

    with gr.Tab("📈 Statistics & Profiling"):
        gr.Markdown("## Data Profiling & Summary Statistics")

        profile_btn = gr.Button("🔍 Generate Data Profile", variant="primary")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Missing Values Report")
                missing_values = gr.Dataframe(label="Missing Values")

            with gr.Column():
                gr.Markdown("### Numerical Summary")
                numerical_summary = gr.Dataframe(label="Descriptive Statistics")

        gr.Markdown("### Categorical Summary")
        categorical_summary = gr.Textbox(
            label="Categorical Variables",
            lines=10,
            interactive=False
        )

        gr.Markdown("### Correlation Matrix")
        correlation_plot = gr.Plot(label="Correlation Heatmap")

        def generate_profile():
            if app_state.current_df is None:
                return (
                    None, None, "⚠️ No dataset loaded. Please load a dataset first.", None
                )

            try:
                profile = app_state.processor.get_data_profile()

                # Missing values
                missing_df = profile['missing_values']

                # Numerical summary
                num_summary = profile['numerical_summary']

                # Categorical summary - FIXED
                cat_summary = profile['categorical_summary']
                cat_text = ""
                for col, stats in cat_summary.items():
                    cat_text += f"\n**{col}**:\n"
                    cat_text += f"  - Unique values: {stats['unique_count']}\n"

                    # Safe handling of top_value
                    top_val = stats.get('top_value', 'N/A')
                    if pd.isna(top_val):
                        top_val = 'N/A'
                    cat_text += f"  - Most common: {top_val} ({stats['top_value_frequency']} occurrences)\n"

                    # Safe handling of value_counts
                    if stats.get('value_counts'):
                        top_values = list(stats['value_counts'].keys())[:5]
                        cat_text += f"  - Top values: {', '.join(str(v) for v in top_values)}\n"

                if not cat_text:
                    cat_text = "No categorical columns found."

                # Correlation matrix
                corr_matrix = profile['correlation_matrix']

                if not corr_matrix.empty and len(corr_matrix.columns) >= 2:
                    fig = app_state.viz_manager.create_visualization(
                        'correlation',
                        app_state.current_df,
                        backend='matplotlib'
                    )
                else:
                    fig = None

                return missing_df, num_summary, cat_text, fig

            except Exception as e:
                logger.error(f"Error generating profile: {e}")
                import traceback
                traceback.print_exc()
                return None, None, f"❌ Error: {str(e)}", None

        profile_btn.click(
            fn=generate_profile,
            outputs=[missing_values, numerical_summary, categorical_summary, correlation_plot]
        )

    return profile_btn, missing_values, numerical_summary, categorical_summary, correlation_plot


# ============================================================================
# TAB 3: FILTER & EXPLORE
# ============================================================================

def create_filter_tab():
    """Create interactive filtering tab."""

    with gr.Tab("🔍 Filter & Explore"):
        gr.Markdown("## Interactive Data Filtering")
        gr.Markdown("Apply filters to narrow down your data for analysis")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Filter Controls")

                filter_type = gr.Radio(
                    choices=["Numerical Range", "Categorical Values", "Date Range"],
                    label="Filter Type",
                    value="Numerical Range"
                )

                column_select = gr.Dropdown(
                    label="Select Column",
                    choices=[],
                    interactive=True
                )

                # Numerical filters
                with gr.Group(visible=True) as numerical_group:
                    min_value = gr.Number(label="Minimum Value")
                    max_value = gr.Number(label="Maximum Value")

                # Categorical filters
                with gr.Group(visible=False) as categorical_group:
                    category_select = gr.CheckboxGroup(
                        label="Select Values",
                        choices=[]
                    )

                # Date filters
                with gr.Group(visible=False) as date_group:
                    start_date = gr.Textbox(label="Start Date (YYYY-MM-DD)")
                    end_date = gr.Textbox(label="End Date (YYYY-MM-DD)")

                add_filter_btn = gr.Button("➕ Add Filter", variant="primary")
                clear_filters_btn = gr.Button("🗑️ Clear All Filters", variant="secondary")

            with gr.Column(scale=2):
                filter_status = gr.Textbox(
                    label="Active Filters",
                    value="No filters applied",
                    lines=5,
                    interactive=False
                )

                row_count = gr.Textbox(
                    label="Filtered Row Count",
                    value="0 rows",
                    interactive=False
                )

                filtered_preview = gr.Dataframe(
                    label="Filtered Data Preview",
                    interactive=False
                )

        def update_column_choices(filter_type_value):
            if app_state.current_df is None:
                return gr.Dropdown(choices=[]), gr.Group(visible=False), gr.Group(visible=False), gr.Group(visible=False)

            col_types = get_column_types(app_state.current_df)

            if filter_type_value == "Numerical Range":
                choices = col_types['numerical']
                return (
                    gr.Dropdown(choices=choices),
                    gr.Group(visible=True),
                    gr.Group(visible=False),
                    gr.Group(visible=False)
                )
            elif filter_type_value == "Categorical Values":
                choices = col_types['categorical']
                return (
                    gr.Dropdown(choices=choices),
                    gr.Group(visible=False),
                    gr.Group(visible=True),
                    gr.Group(visible=False)
                )
            else:  # Date Range
                choices = col_types['datetime']
                return (
                    gr.Dropdown(choices=choices),
                    gr.Group(visible=False),
                    gr.Group(visible=False),
                    gr.Group(visible=True)
                )

        def update_category_choices(column):
            if app_state.current_df is None or not column:
                return gr.CheckboxGroup(choices=[])

            unique_values = app_state.current_df[column].dropna().unique().tolist()
            return gr.CheckboxGroup(choices=unique_values[:50])  # Limit to 50 for performance

        def add_filter(filter_type_value, column, min_val, max_val, categories, start, end):
            if app_state.current_df is None:
                return "⚠️ No dataset loaded", "0 rows", None

            if not column:
                return "⚠️ Please select a column", f"{len(app_state.filtered_df)} rows", app_state.filtered_df.head(100)

            # Create filter configuration
            filter_config = {'column': column}

            if filter_type_value == "Numerical Range":
                filter_config['type'] = 'numerical'
                filter_config['min_val'] = min_val
                filter_config['max_val'] = max_val
            elif filter_type_value == "Categorical Values":
                filter_config['type'] = 'categorical'
                filter_config['values'] = categories if categories else []
            else:  # Date Range
                filter_config['type'] = 'date'
                filter_config['start_date'] = start if start else None
                filter_config['end_date'] = end if end else None

            # Add to active filters
            app_state.active_filters.append(filter_config)

            # Apply all filters
            filtered_df = app_state.apply_filters(app_state.active_filters)

            # Generate status message
            status = "**Active Filters:**\n"
            for i, f in enumerate(app_state.active_filters, 1):
                status += f"{i}. {f['column']} ({f['type']})\n"

            row_info = f"{len(filtered_df)} rows (filtered from {len(app_state.current_df)})"

            return status, row_info, filtered_df.head(100)

        def clear_all_filters():
            if app_state.current_df is None:
                return "No filters applied", "0 rows", None

            app_state.reset_filters()
            row_info = f"{len(app_state.current_df)} rows"

            return "No filters applied", row_info, app_state.current_df.head(100)

        # Event handlers
        filter_type.change(
            fn=update_column_choices,
            inputs=[filter_type],
            outputs=[column_select, numerical_group, categorical_group, date_group]
        )

        column_select.change(
            fn=update_category_choices,
            inputs=[column_select],
            outputs=[category_select]
        )

        add_filter_btn.click(
            fn=add_filter,
            inputs=[filter_type, column_select, min_value, max_value, category_select, start_date, end_date],
            outputs=[filter_status, row_count, filtered_preview]
        )

        clear_filters_btn.click(
            fn=clear_all_filters,
            outputs=[filter_status, row_count, filtered_preview]
        )

    return (filter_type, column_select, filter_status, row_count, filtered_preview,
            add_filter_btn, clear_filters_btn)


# ============================================================================
# TAB 4: VISUALIZATIONS
# ============================================================================

def create_visualization_tab():
    """Create visualization tab with smart recommendations."""

    with gr.Tab("📉 Visualizations"):
        gr.Markdown("## Create Visualizations")

        # Smart Recommendations Section
        with gr.Accordion("🎯 Smart Recommendations", open=True):
            recommend_btn = gr.Button("💡 Get Visualization Recommendations", variant="primary")
            recommendations_output = gr.Markdown(value="Click the button to get recommendations")

            # Dynamic recommendation buttons
            with gr.Row(visible=False) as rec_buttons_row:
                rec_btn_1 = gr.Button("", visible=False, variant="secondary", scale=1)
                rec_btn_2 = gr.Button("", visible=False, variant="secondary", scale=1)
                rec_btn_3 = gr.Button("", visible=False, variant="secondary", scale=1)

            rec_viz_output = gr.Plot(label="Recommended Visualization", visible=False)
            rec_status = gr.Textbox(label="Status", visible=False, interactive=False)

            def get_recommendations():
                if app_state.filtered_df is None or app_state.filtered_df.empty:
                    return "⚠️ No data available. Please load a dataset first.", gr.Row(visible=False), "", "", "", gr.Plot(visible=False), gr.Textbox(visible=False)

                recommender = SmartVisualizationRecommender()
                analysis = recommender.analyze_dataset(app_state.filtered_df)
                app_state.current_recommendations = analysis['recommendations']

                output = "## 🎯 Recommended Visualizations\n\n"
                output += analysis['summary'] + "\n\n"

                output += "### Click below to create recommended visualizations:\n\n"

                # Prepare button labels
                btn_labels = ["", "", ""]
                for i, rec in enumerate(analysis['recommendations'][:3]):
                    priority_emoji = "🔴" if rec['priority'] == 'high' else "🟡"
                    btn_labels[i] = f"{priority_emoji} Create {rec['type'].replace('_', ' ').title()}"

                return (
                    output,
                    gr.Row(visible=True),
                    gr.Button(value=btn_labels[0], visible=True) if btn_labels[0] else gr.Button(visible=False),
                    gr.Button(value=btn_labels[1], visible=True) if btn_labels[1] else gr.Button(visible=False),
                    gr.Button(value=btn_labels[2], visible=True) if btn_labels[2] else gr.Button(visible=False),
                    gr.Plot(visible=False),
                    gr.Textbox(visible=False)
                )

            def create_recommended_viz(rec_index):
                if app_state.current_recommendations is None or rec_index >= len(app_state.current_recommendations):
                    return None, "⚠️ No recommendation available"

                rec = app_state.current_recommendations[rec_index]

                try:
                    if rec['type'] == 'time_series':
                        params = rec['suggested_params']
                        fig = app_state.viz_manager.create_visualization(
                            'time_series',
                            app_state.filtered_df,
                            date_column=params['date_column'],
                            value_column=params['value_column'],
                            aggregation=params['aggregation'],
                            backend='matplotlib'
                        )
                        status = f"✅ Created recommended time series plot"

                    elif rec['type'] == 'correlation':
                        fig = app_state.viz_manager.create_visualization(
                            'correlation',
                            app_state.filtered_df,
                            backend='matplotlib'
                        )
                        status = "✅ Created recommended correlation heatmap"

                    elif rec['type'] == 'category':
                        params = rec['suggested_params']
                        fig = app_state.viz_manager.create_visualization(
                            'category',
                            app_state.filtered_df,
                            column=params['column'],
                            plot_type=params['plot_type'],
                            backend='matplotlib'
                        )
                        status = f"✅ Created recommended category plot"

                    elif rec['type'] == 'distribution':
                        params = rec['suggested_params']
                        fig = app_state.viz_manager.create_visualization(
                            'distribution',
                            app_state.filtered_df,
                            column=params['column'],
                            plot_type=params['plot_type'],
                            backend='matplotlib'
                        )
                        status = "✅ Created recommended distribution plot"

                    elif rec['type'] == 'scatter':
                        params = rec['suggested_params']
                        fig = app_state.viz_manager.create_visualization(
                            'scatter',
                            app_state.filtered_df,
                            x_column=params['x_column'],
                            y_column=params['y_column'],
                            backend='matplotlib'
                        )
                        status = "✅ Created recommended scatter plot"
                    else:
                        return None, "❌ Unknown recommendation type"

                    return gr.Plot(value=fig, visible=True), gr.Textbox(value=status, visible=True)

                except Exception as e:
                    logger.error(f"Error creating recommended visualization: {e}")
                    return None, gr.Textbox(value=f"❌ Error: {str(e)}", visible=True)

            recommend_btn.click(
                fn=get_recommendations,
                outputs=[recommendations_output, rec_buttons_row, rec_btn_1, rec_btn_2, rec_btn_3, rec_viz_output, rec_status]
            )

            rec_btn_1.click(
                fn=lambda: create_recommended_viz(0),
                outputs=[rec_viz_output, rec_status]
            )

            rec_btn_2.click(
                fn=lambda: create_recommended_viz(1),
                outputs=[rec_viz_output, rec_status]
            )

            rec_btn_3.click(
                fn=lambda: create_recommended_viz(2),
                outputs=[rec_viz_output, rec_status]
            )

        gr.Markdown("---")
        gr.Markdown("### Create Custom Visualization")

        with gr.Row():
            with gr.Column(scale=1):
                viz_type = gr.Dropdown(
                    label="Visualization Type",
                    choices=[
                        "Time Series",
                        "Distribution (Histogram)",
                        "Distribution (Box Plot)",
                        "Category (Bar Chart)",
                        "Category (Pie Chart)",
                        "Scatter Plot",
                        "Correlation Heatmap"
                    ],
                    value="Time Series"
                )

                # Dynamic parameter inputs
                with gr.Group() as time_series_group:
                    ts_date_col = gr.Dropdown(label="Date Column", choices=[])
                    ts_value_col = gr.Dropdown(label="Value Column", choices=[])
                    ts_agg = gr.Dropdown(
                        label="Aggregation",
                        choices=["sum", "mean", "count", "median"],
                        value="sum"
                    )

                with gr.Group(visible=False) as distribution_group:
                    dist_col = gr.Dropdown(label="Column", choices=[])
                    dist_bins = gr.Slider(label="Number of Bins", minimum=10, maximum=100, value=30, step=5)

                with gr.Group(visible=False) as category_group:
                    cat_col = gr.Dropdown(label="Category Column", choices=[])
                    cat_value_col = gr.Dropdown(label="Value Column (optional)", choices=[])
                    cat_agg = gr.Dropdown(
                        label="Aggregation",
                        choices=["count", "sum", "mean", "median"],
                        value="count"
                    )
                    cat_top_n = gr.Slider(label="Top N Categories", minimum=5, maximum=20, value=10, step=1)

                with gr.Group(visible=False) as scatter_group:
                    scatter_x = gr.Dropdown(label="X Column", choices=[])
                    scatter_y = gr.Dropdown(label="Y Column", choices=[])
                    scatter_color = gr.Dropdown(label="Color by (optional)", choices=[])
                    scatter_trend = gr.Checkbox(label="Show Trend Line", value=False)

                with gr.Group(visible=False) as correlation_group:
                    corr_method = gr.Dropdown(
                        label="Correlation Method",
                        choices=["pearson", "spearman", "kendall"],
                        value="pearson"
                    )

                create_viz_btn = gr.Button("📊 Create Visualization", variant="primary")

            with gr.Column(scale=2):
                viz_output = gr.Plot(label="Visualization")
                viz_status = gr.Textbox(label="Status", lines=2, interactive=False)

        def update_viz_controls(viz_type_value):
            if app_state.filtered_df is None:
                return [gr.Group(visible=False)] * 5 + [gr.Dropdown(choices=[])] * 8

            col_types = get_column_types(app_state.filtered_df)

            # FIXED: Return format with value=None to force refresh
            # [5 Groups] + [8 Dropdowns]
            # Groups: time_series_group, distribution_group, category_group, scatter_group, correlation_group
            # Dropdowns: ts_date_col, ts_value_col, dist_col, cat_col, cat_value_col, scatter_x, scatter_y, scatter_color

            if viz_type_value == "Time Series":
                return (
                    gr.Group(visible=True),   # time_series_group
                    gr.Group(visible=False),  # distribution_group
                    gr.Group(visible=False),  # category_group
                    gr.Group(visible=False),  # scatter_group
                    gr.Group(visible=False),  # correlation_group
                    gr.Dropdown(choices=col_types['datetime'], value=None),      # ts_date_col
                    gr.Dropdown(choices=col_types['numerical'], value=None),     # ts_value_col
                    gr.Dropdown(choices=col_types['numerical'], value=None),     # dist_col
                    gr.Dropdown(choices=col_types['categorical'], value=None),   # cat_col
                    gr.Dropdown(choices=col_types['numerical'], value=None),     # cat_value_col
                    gr.Dropdown(choices=col_types['numerical'], value=None),     # scatter_x
                    gr.Dropdown(choices=col_types['numerical'], value=None),     # scatter_y
                    gr.Dropdown(choices=col_types['categorical'] + col_types['numerical'], value=None)  # scatter_color
                )

            elif "Distribution" in viz_type_value:
                return (
                    gr.Group(visible=False),
                    gr.Group(visible=True),
                    gr.Group(visible=False),
                    gr.Group(visible=False),
                    gr.Group(visible=False),
                    gr.Dropdown(choices=col_types['datetime'], value=None),
                    gr.Dropdown(choices=col_types['numerical'], value=None),
                    gr.Dropdown(choices=col_types['numerical'], value=None),     # dist_col - visible
                    gr.Dropdown(choices=col_types['categorical'], value=None),
                    gr.Dropdown(choices=col_types['numerical'], value=None),
                    gr.Dropdown(choices=col_types['numerical'], value=None),
                    gr.Dropdown(choices=col_types['numerical'], value=None),
                    gr.Dropdown(choices=col_types['categorical'] + col_types['numerical'], value=None)
                )

            elif "Category" in viz_type_value:
                return (
                    gr.Group(visible=False),
                    gr.Group(visible=False),
                    gr.Group(visible=True),
                    gr.Group(visible=False),
                    gr.Group(visible=False),
                    gr.Dropdown(choices=col_types['datetime'], value=None),
                    gr.Dropdown(choices=col_types['numerical'], value=None),
                    gr.Dropdown(choices=col_types['numerical'], value=None),
                    gr.Dropdown(choices=col_types['categorical'], value=None),   # cat_col - visible
                    gr.Dropdown(choices=col_types['numerical'], value=None),     # cat_value_col - visible
                    gr.Dropdown(choices=col_types['numerical'], value=None),
                    gr.Dropdown(choices=col_types['numerical'], value=None),
                    gr.Dropdown(choices=col_types['categorical'] + col_types['numerical'], value=None)
                )

            elif viz_type_value == "Scatter Plot":
                return (
                    gr.Group(visible=False),
                    gr.Group(visible=False),
                    gr.Group(visible=False),
                    gr.Group(visible=True),
                    gr.Group(visible=False),
                    gr.Dropdown(choices=col_types['datetime'], value=None),
                    gr.Dropdown(choices=col_types['numerical'], value=None),
                    gr.Dropdown(choices=col_types['numerical'], value=None),
                    gr.Dropdown(choices=col_types['categorical'], value=None),
                    gr.Dropdown(choices=col_types['numerical'], value=None),
                    gr.Dropdown(choices=col_types['numerical'], value=None),     # scatter_x - visible
                    gr.Dropdown(choices=col_types['numerical'], value=None),     # scatter_y - visible
                    gr.Dropdown(choices=col_types['categorical'] + col_types['numerical'], value=None)  # scatter_color - visible
                )

            else:  # Correlation Heatmap
                return (
                    gr.Group(visible=False),
                    gr.Group(visible=False),
                    gr.Group(visible=False),
                    gr.Group(visible=False),
                    gr.Group(visible=True),
                    gr.Dropdown(choices=col_types['datetime'], value=None),
                    gr.Dropdown(choices=col_types['numerical'], value=None),
                    gr.Dropdown(choices=col_types['numerical'], value=None),
                    gr.Dropdown(choices=col_types['categorical'], value=None),
                    gr.Dropdown(choices=col_types['numerical'], value=None),
                    gr.Dropdown(choices=col_types['numerical'], value=None),
                    gr.Dropdown(choices=col_types['numerical'], value=None),
                    gr.Dropdown(choices=col_types['categorical'] + col_types['numerical'], value=None)
                )

        def create_visualization(viz_type_value, date_col, value_col, agg,
                                dist_column, bins, cat_column, cat_value, cat_aggregation, top_n,
                                x_col, y_col, color_col, trend, corr_method_value):
            if app_state.filtered_df is None or app_state.filtered_df.empty:
                return None, "⚠️ No data available"

            try:
                if viz_type_value == "Time Series":
                    if not date_col or not value_col:
                        return None, "⚠️ Please select date and value columns"

                    fig = app_state.viz_manager.create_visualization(
                        'time_series',
                        app_state.filtered_df,
                        date_column=date_col,
                        value_column=value_col,
                        aggregation=agg,
                        backend='matplotlib'
                    )
                    status = f"✅ Created time series plot: {value_col} over {date_col}"

                elif "Distribution" in viz_type_value:
                    if not dist_column:
                        return None, "⚠️ Please select a column"

                    plot_type = 'histogram' if 'Histogram' in viz_type_value else 'box'

                    fig = app_state.viz_manager.create_visualization(
                        'distribution',
                        app_state.filtered_df,
                        column=dist_column,
                        plot_type=plot_type,
                        bins=int(bins),
                        backend='matplotlib'
                    )
                    status = f"✅ Created {plot_type} plot for {dist_column}"

                elif "Category" in viz_type_value:
                    if not cat_column:
                        return None, "⚠️ Please select a category column"

                    plot_type = 'bar' if 'Bar' in viz_type_value else 'pie'

                    fig = app_state.viz_manager.create_visualization(
                        'category',
                        app_state.filtered_df,
                        column=cat_column,
                        value_column=cat_value if cat_value else None,
                        plot_type=plot_type,
                        aggregation=cat_aggregation,
                        top_n=int(top_n),
                        backend='matplotlib'
                    )
                    status = f"✅ Created {plot_type} chart for {cat_column}"

                elif viz_type_value == "Scatter Plot":
                    if not x_col or not y_col:
                        return None, "⚠️ Please select X and Y columns"

                    fig = app_state.viz_manager.create_visualization(
                        'scatter',
                        app_state.filtered_df,
                        x_column=x_col,
                        y_column=y_col,
                        color_column=color_col if color_col else None,
                        show_trend=trend,
                        backend='matplotlib'
                    )
                    status = f"✅ Created scatter plot: {y_col} vs {x_col}"

                else:  # Correlation Heatmap
                    fig = app_state.viz_manager.create_visualization(
                        'correlation',
                        app_state.filtered_df,
                        method=corr_method_value,
                        backend='matplotlib'
                    )
                    status = "✅ Created correlation heatmap"

                return fig, status

            except Exception as e:
                logger.error(f"Error creating visualization: {e}")
                import traceback
                traceback.print_exc()
                return None, f"❌ Error: {str(e)}"

        viz_type.change(
            fn=update_viz_controls,
            inputs=[viz_type],
            outputs=[
                time_series_group, distribution_group, category_group,
                scatter_group, correlation_group,
                ts_date_col, ts_value_col, dist_col, cat_col, cat_value_col,
                scatter_x, scatter_y, scatter_color
            ]
        )

        create_viz_btn.click(
            fn=create_visualization,
            inputs=[
                viz_type, ts_date_col, ts_value_col, ts_agg,
                dist_col, dist_bins, cat_col, cat_value_col, cat_agg, cat_top_n,
                scatter_x, scatter_y, scatter_color, scatter_trend, corr_method
            ],
            outputs=[viz_output, viz_status]
        )

    return (viz_type, recommend_btn, recommendations_output, rec_buttons_row,
            rec_btn_1, rec_btn_2, rec_btn_3, rec_viz_output, rec_status,
            viz_output, viz_status, create_viz_btn)


# ============================================================================
# TAB 5: INSIGHTS
# ============================================================================

def create_insights_tab():
    """Create automated insights tab."""

    with gr.Tab("💡 Insights"):
        gr.Markdown("## Automated Insights")
        gr.Markdown("Generate intelligent insights from your data automatically")

        with gr.Row():
            generate_all_btn = gr.Button("🚀 Generate All Insights", variant="primary", scale=2)
            generate_custom_btn = gr.Button("⚙️ Generate Custom Insight", variant="secondary", scale=1)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Custom Insight Options")

                insight_type = gr.Dropdown(
                    label="Insight Type",
                    choices=[
                        "Top/Bottom Performers",
                        "Trend Analysis",
                        "Anomaly Detection",
                        "Distribution Analysis",
                        "Correlation Analysis"
                    ],
                    value="Top/Bottom Performers"
                )

                insight_column = gr.Dropdown(label="Select Column", choices=[])
                insight_column2 = gr.Dropdown(label="Second Column (for trends)", choices=[], visible=False)

            with gr.Column(scale=2):
                insights_output = gr.Textbox(
                    label="Insights Report",
                    lines=20,
                    interactive=False
                )

        def update_insight_columns(insight_type_value):
            if app_state.filtered_df is None:
                return gr.Dropdown(choices=[]), gr.Dropdown(choices=[], visible=False)

            col_types = get_column_types(app_state.filtered_df)

            if insight_type_value == "Trend Analysis":
                return (
                    gr.Dropdown(choices=col_types['datetime']),
                    gr.Dropdown(choices=col_types['numerical'], visible=True)
                )
            else:
                all_cols = col_types['numerical'] + col_types['categorical']
                return (
                    gr.Dropdown(choices=all_cols),
                    gr.Dropdown(choices=[], visible=False)
                )

        def generate_all_insights():
            if app_state.filtered_df is None or app_state.filtered_df.empty:
                return "⚠️ No data available. Please load a dataset first."

            try:
                insights = app_state.insight_manager.generate_all_insights(app_state.filtered_df)
                report = app_state.insight_manager.format_insight_report(insights)
                return report
            except Exception as e:
                logger.error(f"Error generating insights: {e}")
                return f"❌ Error generating insights: {str(e)}"

        def generate_custom_insight(insight_type_value, col1, col2):
            if app_state.filtered_df is None or app_state.filtered_df.empty:
                return "⚠️ No data available"

            if not col1:
                return "⚠️ Please select a column"

            try:
                if insight_type_value == "Top/Bottom Performers":
                    insight = app_state.insight_manager.generate_insight(
                        'top_bottom',
                        app_state.filtered_df,
                        column=col1
                    )

                elif insight_type_value == "Trend Analysis":
                    if not col2:
                        return "⚠️ Please select both date and value columns"

                    insight = app_state.insight_manager.generate_insight(
                        'trend',
                        app_state.filtered_df,
                        date_column=col1,
                        value_column=col2
                    )

                elif insight_type_value == "Anomaly Detection":
                    insight = app_state.insight_manager.generate_insight(
                        'anomaly',
                        app_state.filtered_df,
                        column=col1
                    )

                elif insight_type_value == "Distribution Analysis":
                    insight = app_state.insight_manager.generate_insight(
                        'distribution',
                        app_state.filtered_df,
                        column=col1
                    )

                else:  # Correlation Analysis
                    insight = app_state.insight_manager.generate_insight(
                        'correlation',
                        app_state.filtered_df
                    )

                # Format single insight
                report = f"## {insight_type_value}\n\n"
                report += f"**Summary**: {insight.get('summary', 'No summary available')}\n\n"

                return report

            except Exception as e:
                logger.error(f"Error generating custom insight: {e}")
                return f"❌ Error: {str(e)}"

        insight_type.change(
            fn=update_insight_columns,
            inputs=[insight_type],
            outputs=[insight_column, insight_column2]
        )

        generate_all_btn.click(
            fn=generate_all_insights,
            outputs=[insights_output]
        )

        generate_custom_btn.click(
            fn=generate_custom_insight,
            inputs=[insight_type, insight_column, insight_column2],
            outputs=[insights_output]
        )

    return generate_all_btn, insight_type, insights_output


# ============================================================================
# TAB 6: EXPORT
# ============================================================================

def create_export_tab():
    """Create data export tab."""

    with gr.Tab("💾 Export"):
        gr.Markdown("## Export Data & Visualizations")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Export Filtered Data")
                export_format = gr.Radio(
                    choices=["CSV", "Excel"],
                    label="Export Format",
                    value="CSV"
                )

                export_data_btn = gr.Button("📥 Export Data", variant="primary")
                export_file = gr.File(label="Download File")
                export_status = gr.Textbox(label="Status", lines=2, interactive=False)

            with gr.Column():
                gr.Markdown("### Export Instructions")
                gr.Markdown("""
                **Export Your Data:**
                1. Apply any filters you want in the Filter tab
                2. Select your preferred export format
                3. Click 'Export Data' to download
                
                **Note:** The export will include only the filtered data.
                """)

        def export_data(format_choice):
            if app_state.filtered_df is None or app_state.filtered_df.empty:
                return None, "⚠️ No data to export"

            try:
                import tempfile

                if format_choice == "CSV":
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
                    exporter = CSVExporter()
                    exporter.export(app_state.filtered_df, temp_file.name)
                    status = f"✅ Exported {len(app_state.filtered_df)} rows to CSV"
                else:  # Excel
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
                    exporter = ExcelExporter()
                    exporter.export(app_state.filtered_df, temp_file.name)
                    status = f"✅ Exported {len(app_state.filtered_df)} rows to Excel"

                return temp_file.name, status

            except Exception as e:
                logger.error(f"Error exporting data: {e}")
                return None, f"❌ Error: {str(e)}"

        export_data_btn.click(
            fn=export_data,
            inputs=[export_format],
            outputs=[export_file, export_status]
        )

    return export_data_btn, export_file, export_status


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def create_dashboard():
    """Create the main Business Intelligence Dashboard."""

    with gr.Blocks(title="Business Intelligence Dashboard") as demo:

        # Header
        gr.Markdown("""
        # 📊 Business Intelligence Dashboard
        ### Explore, Analyze, and Extract Insights from Your Data
        
        **Features:** Smart Visualizations | Automated Insights | Interactive Filtering | Data Export
        """)

        # Create all tabs and capture their components
        with gr.Tabs():
            # Tab 1: Dataset Selection
            (dataset_dropdown, status_box, dataset_info, data_preview,
             load_btn, upload_btn) = create_dataset_tab()

            # Tab 2: Statistics
            (profile_btn, missing_values, numerical_summary,
             categorical_summary, correlation_plot) = create_statistics_tab()

            # Tab 3: Filter
            (filter_type, column_select, filter_status, row_count,
             filtered_preview, add_filter_btn, clear_filters_btn) = create_filter_tab()

            # Tab 4: Visualizations
            (viz_type, recommend_btn, recommendations_output, rec_buttons_row,
             rec_btn_1, rec_btn_2, rec_btn_3, rec_viz_output, rec_status,
             viz_output, viz_status, create_viz_btn) = create_visualization_tab()

            # Tab 5: Insights
            (generate_all_btn, insight_type, insights_output) = create_insights_tab()

            # Tab 6: Export
            (export_btn, export_file, export_status_export) = create_export_tab()

        # Footer
        gr.Markdown("""
        ---
        **Business Intelligence Dashboard** | Built with Gradio, Pandas, Matplotlib, and Plotly
        
        *Tip: Start by loading a dataset from the Dataset Selection tab!*
        """)

        # Connect load button to reset all tabs
        def load_and_reset(dataset_name):
            # Load dataset
            if not dataset_name:
                return (
                    None, "⚠️ Please select a dataset", "", None,
                    None, None, "", None,
                    "No filters applied", "0 rows", None,
                    "Click the button to get recommendations",
                    None, None,
                    ""
                )

            df, status = app_state.load_dataset(dataset_name)

            if df is not None:
                info = f"📊 **Dataset**: {dataset_name}\n"
                info += f"📏 **Shape**: {df.shape[0]} rows × {df.shape[1]} columns\n"
                info += f"💾 **Memory**: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB\n\n"
                info += f"**Column Types**:\n"

                col_types = get_column_types(df)
                info += f"- Numerical: {len(col_types['numerical'])}\n"
                info += f"- Categorical: {len(col_types['categorical'])}\n"
                info += f"- DateTime: {len(col_types['datetime'])}\n"

                preview = df.head(100)

                return (
                    dataset_name, status, info, preview,
                    None, None, "", None,
                    "No filters applied", "0 rows", None,
                    "Click the button to get recommendations",
                    None, None,
                    ""
                )

            return (
                None, status, "", None,
                None, None, "", None,
                "No filters applied", "0 rows", None,
                "Click the button to get recommendations",
                None, None,
                ""
            )

        load_btn.click(
            fn=load_and_reset,
            inputs=[dataset_dropdown],
            outputs=[
                dataset_dropdown, status_box, dataset_info, data_preview,
                missing_values, numerical_summary, categorical_summary, correlation_plot,
                filter_status, row_count, filtered_preview,
                recommendations_output,
                viz_output, viz_status,
                insights_output
            ]
        )

    return demo


# ============================================================================
# LAUNCH APPLICATION
# ============================================================================

if __name__ == "__main__":
    logger.info("Starting Business Intelligence Dashboard...")

    # Create and launch dashboard
    demo = create_dashboard()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )