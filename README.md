---
title: Business Intelligence Dashboard
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 3.50.2
app_file: app.py
pinned: false
---

# 📊 Business Intelligence Dashboard

A professional, interactive Business Intelligence dashboard built with Gradio that enables non-technical stakeholders to explore and analyze business data.

## 🌟 Features

### 📂 Data Management
- **Pre-loaded Datasets**: Online Retail and Airbnb datasets included
- **Custom Upload**: Support for CSV, Excel (.xlsx, .xls), JSON, and Parquet files (max 50MB)
- **Automatic Data Cleaning**: Handles missing values, type conversions, and duplicate removal
- **Data Validation**: Comprehensive error handling and user-friendly error messages

### 📈 Statistics & Profiling
- **Automated Data Profiling**: Get instant insights into your dataset
- **Numerical Summary**: Mean, median, std deviation, quartiles, min/max
- **Categorical Analysis**: Unique values, value counts, mode
- **Missing Values Report**: Identify data quality issues
- **Correlation Matrix**: Visual correlation heatmap for numerical features

### 🔍 Interactive Filtering
- **Dynamic Filters**: Filter by numerical ranges, categorical values, or date ranges
- **Real-time Updates**: See row counts update as you apply filters
- **Multiple Filters**: Combine multiple filters for precise data exploration
- **Filter Management**: Easy to add, view, and clear filters

### 📉 Smart Visualizations
- **AI-Powered Recommendations**: Get intelligent visualization suggestions based on your data
- **One-Click Creation**: Create recommended visualizations with a single click
- **5 Visualization Types**:
  - Time Series Plots (with aggregation: sum, mean, count, median)
  - Distribution Plots (histogram, box plot)
  - Category Analysis (bar chart, pie chart)
  - Scatter Plots (with color coding and trend lines)
  - Correlation Heatmap
- **Dual Backend**: Supports both Matplotlib and Plotly
- **Customization**: Full control over columns, aggregations, and visual parameters

### 💡 Automated Insights
- **Top/Bottom Performers**: Identify highest and lowest values
- **Trend Analysis**: Detect patterns over time with growth rate and volatility
- **Anomaly Detection**: Find outliers using Z-score or IQR methods
- **Distribution Analysis**: Understand data distributions with skewness and kurtosis
- **Correlation Insights**: Discover strong relationships between variables

### 💾 Export Capabilities
- **Data Export**: Export filtered data as CSV or Excel
- **Visualization Export**: Save charts as PNG images

## 🏗️ Architecture & Design

### SOLID Principles Implementation
- **Single Responsibility**: Each class has one clear purpose
- **Open/Closed**: Extensible through Strategy Pattern without modifying existing code
- **Liskov Substitution**: All strategies are interchangeable
- **Interface Segregation**: Specific interfaces for different operations
- **Dependency Inversion**: Depends on abstractions, not concrete implementations

### Design Patterns
- **Strategy Pattern**: Used for data loading, visualizations, and insights
- **Facade Pattern**: DataProcessor provides simple interface to complex operations
- **Factory Pattern**: Dynamic strategy selection based on file type

### Project Structure
```
Business-Intelligence-Dashboard/
├── app.py                      # Main Gradio application with 6 tabs
├── data_processor.py           # Data loading, cleaning, filtering (Strategy Pattern)
├── visualizations.py           # Chart creation with multiple strategies
├── insights.py                 # Automated insight generation
├── utils.py                    # Utility functions and validators
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/                       # Sample datasets
│   ├── Online_Retail.xlsx
│   └── Airbnb.csv
└── tests/                      # Comprehensive test suite
    ├── init.py
    ├── conftest.py
    ├── test_utils.py
    ├── test_data_processor.py
    ├── test_visualizations.py
    └── test_insights.py
```
## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/CR1502/Business-Intelligence-Dashboard.git
cd Business-Intelligence-Dashboard
```

2. **Create a virtual environment**
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
python app.py
```

The dashboard will launch and open in your default browser at `http://localhost:7860`

## 📖 Usage Guide

### 1. Loading Data
- **Option A**: Select "Online Retail" or "Airbnb" from the dropdown
- **Option B**: Upload your own dataset (CSV, Excel, JSON, or Parquet)

### 2. Exploring Statistics
- Navigate to "Statistics & Profiling" tab
- Click "Generate Data Profile" to see comprehensive statistics
- View missing values, numerical summaries, and correlation matrix

### 3. Filtering Data
- Go to "Filter & Explore" tab
- Select filter type (Numerical, Categorical, or Date)
- Choose column and set filter criteria
- Click "Add Filter" and see real-time updates

### 4. Creating Visualizations
- Navigate to "Visualizations" tab
- **Smart Recommendations**: Click "Get Visualization Recommendations" for AI-powered suggestions
- **Custom Visualizations**: Select visualization type and configure parameters
- Supported charts: Time Series, Distribution, Category, Scatter, Correlation

### 5. Generating Insights
- Go to "Insights" tab
- Click "Generate All Insights" for automated analysis
- Or select specific insight type for targeted analysis

### 6. Exporting Results
- Navigate to "Export" tab
- Choose format (CSV or Excel)
- Click "Export Data" to download filtered dataset

## 🧪 Testing

Run the comprehensive test suite:
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_utils.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

Test coverage includes:
- **180+ test cases** across all modules
- Unit tests for all functions and classes
- Strategy Pattern implementation tests
- Edge case and error handling tests

## 🛠️ Technologies Used

- **Gradio**: Web interface and interactive components
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Static visualizations
- **Plotly**: Interactive visualizations
- **Python 3.10+**: Core programming language

## 📊 Sample Datasets

### Online Retail Dataset
- **8 columns**: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
- **Use case**: E-commerce sales analysis, product trends, customer analysis

### Airbnb Dataset
- **26 columns**: Including price, location, room type, reviews, availability
- **Use case**: Pricing analysis, location trends, booking patterns

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update README.md for significant changes


## 👨‍💻 Author

**Craig Roberts**


## 🙏 Acknowledgments

- Northeastern University - CS5130 Course (Prof Lino)
- Dataset sources: UCI ML Repository, Kaggle

## ⚡ Performance Notes

- Handles datasets up to 50MB efficiently
- Optimized for 1,000-10,000 rows
- Tested with datasets containing 100+ columns
- Real-time filtering with sub-second response times

## 🐛 Known Issues

- Large datasets (>100MB) may cause memory issues
- Some complex visualizations may take time to render
- Browser storage not available (by design for security)
---
