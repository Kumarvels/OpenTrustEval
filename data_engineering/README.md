# OpenTrustEval - Trust Scoring System Dashboard

## 🚀 Overview

OpenTrustEval is a comprehensive trustworthy AI evaluation framework that provides advanced data quality assessment, trust scoring, and benchmarking capabilities. The system includes a powerful Streamlit dashboard with multi-source file upload functionality, real-time analytics, and comprehensive reporting.

![Dashboard Screenshot](https://img.shields.io/badge/Dashboard-Streamlit-blue)
![File Upload](https://img.shields.io/badge/File%20Upload-Multi%20Source-green)
![Trust Scoring](https://img.shields.io/badge/Trust%20Scoring-Advanced-orange)

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Dashboard Guide](#-dashboard-guide)
- [File Upload System](#-file-upload-system)
- [Trust Scoring](#-trust-scoring)
- [Cleanlab Comparison](#-cleanlab-comparison)
- [Command Line Interface](#-command-line-interface)
- [API Reference](#-api-reference)
- [Examples](#-examples)
- [Troubleshooting](#-troubleshooting)

## ✨ Features

### 🔍 **Core Trust Scoring**
- **Advanced Trust Scoring Engine**: Multi-method trust assessment
- **Quality Metrics**: Comprehensive data quality evaluation
- **Statistical Analysis**: Robust statistical validation
- **Real-time Processing**: Instant trust score calculation

### 📁 **Multi-Source File Upload**
- **Local File Upload**: Drag & drop from your computer
- **Amazon S3**: Direct download from S3 buckets
- **Google Drive**: Upload from Google Drive files
- **Google Cloud Storage**: Download from GCS buckets
- **Local Path**: Copy from local file system
- **External Drives**: Support for external storage devices

### 📊 **Supported File Formats**
- **Data Formats**: CSV, JSON, Excel, Parquet, HDF5
- **Advanced Formats**: Pickle, Feather, Stata, SAS, SPSS
- **Structured Data**: XML, YAML
- **Auto-detection**: Automatic format recognition

### 🎯 **Dashboard Features**
- **Real-time Analytics**: Live data visualization
- **Interactive Charts**: Plotly-powered visualizations
- **SQL Query Interface**: Direct database queries
- **Report Generation**: Automated report creation
- **Session Management**: Persistent session tracking

### 🔬 **Cleanlab Integration**
- **Benchmarking**: Compare against industry standards
- **Multiple Options**: 4 different Cleanlab scoring methods
- **Statistical Testing**: Statistical comparison analysis
- **Visualization**: Side-by-side comparison charts

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Git (for cloning)

### Basic Installation
```bash
# Clone the repository
git clone https://github.com/your-repo/OpenTrustEval.git
cd OpenTrustEval

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# Install core dependencies
pip install -r requirements.txt
```

### Advanced Installation (with Cloud Storage)
```bash
# Install additional dependencies for cloud storage
pip install -r data_engineering/requirements_file_upload.txt

# Optional: Install specific cloud providers
pip install boto3  # AWS S3
pip install google-cloud-storage  # Google Cloud Storage
pip install pydrive  # Google Drive
```

### Verify Installation
```bash
# Test the installation
cd data_engineering
python test_file_upload.py
```

## 🚀 Quick Start

### 1. Launch the Dashboard
```bash
cd data_engineering
streamlit run trust_scoring_dashboard.py
```

The dashboard will be available at:
- **Local**: http://localhost:8501
- **Network**: http://your-ip:8501

### 2. Upload Your First File
1. Navigate to the **"🔍 Trust Scoring"** page
2. Select **"Local File Upload"** from the upload method dropdown
3. Click **"Browse files"** and select your dataset
4. The system will automatically detect the format and validate the file
5. Click **"Calculate Trust Score"** to analyze your data

### 3. View Results
- Trust scores are displayed in real-time
- Quality metrics are automatically calculated
- Visualizations are generated automatically
- Results are logged to the database for historical tracking

## 📊 Dashboard Guide

### 🏠 **Overview Page**
The overview page provides a 360-degree view of your trust scoring system:

#### Key Metrics
- **Average Trust Score**: Overall system performance
- **Test Success Rate**: System reliability indicator
- **Data Completeness**: Quality of input data
- **Commands Executed**: System usage statistics

#### Recent Activity
- **Recent Trust Scores**: Latest scoring results
- **Recent Test Results**: System test outcomes
- **System Health**: Performance visualizations

#### Visualizations
- **Trust Score Trends**: Time-series analysis
- **Quality Dashboard**: Multi-metric quality overview
- **Test Results Summary**: Success/failure distribution

### 🔍 **Trust Scoring Page**

#### File Upload Section
The file upload section supports multiple data sources:

##### Local File Upload
```
1. Select "Local File Upload"
2. Click "Choose a file"
3. Select your dataset (CSV, JSON, Excel, etc.)
4. System validates format automatically
5. File is saved to uploads directory
```

##### S3 Bucket Upload
```
1. Select "S3 Bucket"
2. Enter bucket name and file key
3. Optionally provide AWS credentials
4. Select AWS region
5. Click "Download from S3"
```

##### Google Drive Upload
```
1. Select "Google Drive"
2. Enter Google Drive file ID
3. Optionally provide credentials path
4. Click "Download from Google Drive"
```

##### Google Cloud Storage Upload
```
1. Select "Google Cloud Storage"
2. Enter bucket name and blob name
3. Optionally provide service account JSON
4. Click "Download from GCS"
```

##### Local Path Upload
```
1. Select "Local Path"
2. Enter full file path
3. Click "Copy from Local Path"
```

#### File Information Display
After upload, the system displays:
- **File Path**: Location of uploaded file
- **File Name**: Original filename
- **File Size**: Size in KB
- **Format**: Detected file format
- **Preview**: Data preview (first 5 rows)

#### Trust Scoring Controls
- **Dataset Path**: Auto-filled from uploaded file
- **Scoring Method**: Choose from ensemble, robust, uncertainty
- **Calculate Trust Score**: Run trust scoring analysis
- **Quality Assessment**: Perform quality evaluation

#### Cleanlab Comparison Section
Compare your trust scores against Cleanlab benchmarks:

##### Data Format Selection
- **CSV**: Comma-separated values
- **JSON**: JavaScript Object Notation
- **Excel**: Microsoft Excel files
- **Parquet**: Columnar storage format
- **HDF5**: Hierarchical Data Format
- **DataFrame**: Pandas DataFrame format

##### Cleanlab Options
- **Option 1**: Basic label quality score
- **Option 2**: Confidence-based score
- **Option 3**: Uncertainty-based score
- **Option 4**: Ensemble Cleanlab score

##### Comparison Methods
- **Side-by-Side**: Direct score comparison
- **Difference Analysis**: Score difference analysis
- **Correlation Analysis**: Correlation between scores
- **Statistical Test**: Statistical significance testing

##### Visualization Types
- **Table**: Tabular comparison results
- **Chart**: Graphical visualization
- **Both**: Combined table and chart view

#### Trust Score History
- **Filter by Method**: Select specific scoring methods
- **Date Range**: Filter by time period
- **Data Table**: Historical trust scores
- **Distribution Chart**: Trust score distribution

### 📈 **Analytics Page**
Advanced analytics and insights:

#### Performance Metrics
- **Trust Score Trends**: Time-series analysis
- **Quality Metrics**: Data quality over time
- **System Performance**: Resource utilization
- **Error Analysis**: Failure pattern analysis

#### Custom Analytics
- **SQL Queries**: Direct database access
- **Custom Visualizations**: User-defined charts
- **Export Capabilities**: Data export options

### ⚡ **Commands Page**
Execute system commands and operations:

#### Available Commands
- **calculate_trust_score**: Calculate trust score for dataset
- **quality_assessment**: Perform quality assessment
- **dataset_validation**: Validate dataset structure
- **batch_test_suite**: Run comprehensive test suite

#### Command Execution
1. Select command from dropdown
2. Provide dataset path (if required)
3. Click "Execute Command"
4. View results and logs

### 🧪 **Testing Page**
System testing and validation:

#### Test Types
- **Unit Tests**: Individual component testing
- **Integration Tests**: System integration testing
- **Performance Tests**: Load and stress testing
- **Regression Tests**: Automated regression testing

#### Test Execution
- **Run Individual Tests**: Execute specific tests
- **Run Test Suite**: Execute all tests
- **View Test Results**: Detailed test outcomes
- **Test History**: Historical test results

### 📋 **Reports Page**
Generate comprehensive reports:

#### Report Types
- **Trust Score Report**: Detailed trust scoring analysis
- **Quality Report**: Data quality assessment
- **Performance Report**: System performance analysis
- **Comparison Report**: Cleanlab comparison results

#### Report Generation
1. Select report type
2. Choose date range
3. Select datasets (if applicable)
4. Click "Generate Report"
5. Download or view report

### 🗄️ **SQL Query Page**
Direct database access and querying:

#### Query Interface
- **Query Editor**: SQL query input
- **Query History**: Previous queries
- **Results Display**: Query results table
- **Export Results**: Export query results

#### Example Queries
```sql
-- Get recent trust scores
SELECT * FROM trust_scores 
ORDER BY timestamp DESC 
LIMIT 10;

-- Get quality metrics
SELECT dataset_name, AVG(data_completeness) as avg_completeness
FROM quality_metrics 
GROUP BY dataset_name;

-- Get system performance
SELECT metric_name, AVG(metric_value) as avg_value
FROM system_metrics 
GROUP BY metric_name;
```

## 📁 File Upload System

### Supported File Formats

#### Data Formats
| Format | Extension | Description | Use Case |
|--------|-----------|-------------|----------|
| CSV | .csv | Comma-separated values | Tabular data, spreadsheets |
| JSON | .json | JavaScript Object Notation | Structured data, APIs |
| Excel | .xlsx, .xls | Microsoft Excel | Business data, reports |
| Parquet | .parquet | Columnar storage | Big data, analytics |
| HDF5 | .h5, .hdf5 | Hierarchical Data Format | Scientific data, arrays |
| Pickle | .pkl, .pickle | Python serialization | Python objects, models |
| Feather | .feather | Fast columnar format | Fast I/O operations |
| Stata | .dta | Stata format | Statistical analysis |
| SAS | .sas7bdat | SAS format | Statistical analysis |
| SPSS | .sav | SPSS format | Statistical analysis |
| XML | .xml | Extensible Markup Language | Structured documents |
| YAML | .yaml, .yml | YAML format | Configuration files |

### Upload Methods

#### 1. Local File Upload
**Best for**: Quick testing, small files, local development

```python
# Features
- Drag & drop interface
- Automatic format detection
- File validation
- Preview capability
- Size limit: 200MB (configurable)
```

#### 2. Amazon S3
**Best for**: Cloud storage, large files, production environments

```python
# Configuration
- Bucket name: Your S3 bucket
- File key: Path to file in bucket
- Region: AWS region (us-east-1, us-west-2, etc.)
- Credentials: AWS access key/secret (optional)

# Example
Bucket: my-data-bucket
File Key: datasets/customer_data.csv
Region: us-east-1
```

#### 3. Google Drive
**Best for**: Collaboration, sharing, Google Workspace integration

```python
# Configuration
- File ID: Google Drive file ID
- Credentials: OAuth credentials (optional)

# How to get File ID
1. Right-click file in Google Drive
2. Select "Get link"
3. Copy ID from URL: https://drive.google.com/file/d/FILE_ID/view
```

#### 4. Google Cloud Storage
**Best for**: Enterprise cloud storage, GCP integration

```python
# Configuration
- Bucket name: GCS bucket name
- Blob name: File path in bucket
- Credentials: Service account JSON (optional)

# Example
Bucket: my-gcs-bucket
Blob: data/analytics_dataset.parquet
```

#### 5. Local Path
**Best for**: Existing files, batch processing, automation

```python
# Configuration
- File path: Full path to file

# Example
Path: C:/Users/username/Documents/data.csv
Path: /home/user/datasets/analysis.xlsx
```

### File Validation

#### Automatic Validation
The system automatically validates uploaded files:

```python
# Validation checks
- File format detection
- File size verification
- Data structure validation
- Encoding detection
- Corrupted file detection
```

#### Manual Validation
You can manually validate files:

```python
from data_engineering.trust_scoring_dashboard import TrustScoringDashboard

dashboard = TrustScoringDashboard()
is_valid, format_type = dashboard.validate_file_format("path/to/file.csv")
print(f"Valid: {is_valid}, Format: {format_type}")
```

### Data Preview

#### Automatic Preview
After upload, the system provides:

```python
# Preview information
- Number of rows and columns
- Column names and types
- First 5 rows of data
- Data type detection
- Missing value detection
```

#### Manual Preview
```python
# Load and preview data
df = dashboard.load_data_by_format("path/to/file.csv", "CSV")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(df.head())
```

## 🔍 Trust Scoring

### Scoring Methods

#### 1. Ensemble Method
**Best for**: Robust, reliable scoring

```python
# Features
- Multiple algorithm combination
- Weighted averaging
- Outlier detection
- Confidence intervals
- Uncertainty quantification
```

#### 2. Robust Method
**Best for**: Noisy data, outliers

```python
# Features
- Outlier-resistant algorithms
- Robust statistics
- Median-based methods
- Huber loss functions
- RANSAC-like approaches
```

#### 3. Uncertainty Method
**Best for**: Uncertainty quantification

```python
# Features
- Bayesian approaches
- Monte Carlo methods
- Confidence intervals
- Uncertainty propagation
- Probabilistic scoring
```

### Quality Metrics

#### Core Metrics
```python
# Data Quality Metrics
- Missing Values Ratio: Percentage of missing data
- Duplicate Rows Ratio: Percentage of duplicate rows
- Outlier Ratio: Percentage of outlier values
- Data Completeness: Overall data completeness score
- Data Consistency: Internal consistency score
```

#### Advanced Metrics
```python
# Statistical Metrics
- Distribution Analysis: Data distribution assessment
- Correlation Analysis: Feature correlation analysis
- Statistical Tests: Normality, homogeneity tests
- Entropy Analysis: Information content analysis
- Complexity Metrics: Data complexity assessment
```

### Trust Score Calculation

#### Process Flow
```python
1. Data Loading: Load and validate input data
2. Preprocessing: Handle missing values, outliers
3. Feature Engineering: Create relevant features
4. Model Application: Apply trust scoring models
5. Post-processing: Aggregate and normalize scores
6. Validation: Cross-validate results
7. Reporting: Generate detailed reports
```

#### Score Interpretation
```python
# Score Ranges
0.0 - 0.3: Low trust (Poor data quality)
0.3 - 0.6: Medium trust (Acceptable quality)
0.6 - 0.8: High trust (Good quality)
0.8 - 1.0: Very high trust (Excellent quality)
```

## 🔬 Cleanlab Comparison

### Comparison Methods

#### 1. Side-by-Side Comparison
**Purpose**: Direct score comparison

```python
# Features
- Direct score display
- Absolute difference calculation
- Percentage difference
- Visual comparison charts
```

#### 2. Difference Analysis
**Purpose**: Detailed difference analysis

```python
# Features
- Score difference calculation
- Difference distribution
- Statistical significance
- Effect size analysis
```

#### 3. Correlation Analysis
**Purpose**: Relationship analysis

```python
# Features
- Correlation coefficient calculation
- Scatter plot visualization
- Correlation significance testing
- Trend analysis
```

#### 4. Statistical Test
**Purpose**: Statistical comparison

```python
# Features
- T-test for score comparison
- P-value calculation
- Confidence intervals
- Effect size estimation
```

### Cleanlab Options

#### Option 1: Basic Label Quality
```python
# Method
- Basic label quality assessment
- Simple statistical measures
- Fast computation
- Suitable for initial screening
```

#### Option 2: Confidence-Based
```python
# Method
- Confidence-based scoring
- Uncertainty quantification
- Model confidence assessment
- Reliability estimation
```

#### Option 3: Uncertainty-Based
```python
# Method
- Uncertainty-based scoring
- Bayesian approaches
- Probabilistic assessment
- Risk quantification
```

#### Option 4: Ensemble Cleanlab
```python
# Method
- Ensemble of multiple methods
- Weighted combination
- Robust scoring
- Comprehensive assessment
```

### Visualization Types

#### Table View
```python
# Features
- Tabular data display
- Sortable columns
- Filterable rows
- Export capabilities
```

#### Chart View
```python
# Features
- Interactive charts
- Multiple chart types
- Zoom and pan capabilities
- Export to image
```

#### Combined View
```python
# Features
- Both table and chart
- Synchronized views
- Comprehensive analysis
- Detailed insights
```

## 🖥️ Command Line Interface

### Installation
```bash
# The CLI is available at
data_engineering/scripts/dataset_cli.py
```

### Basic Commands

#### Create Dataset
```bash
# Create from CSV
python dataset_cli.py create --name "my_dataset" --input data.csv --format csv

# Create with schema
python dataset_cli.py create --name "employees" --input employees.csv --format csv --schema schema.json

# Create from JSON
python dataset_cli.py create --name "api_data" --input api_response.json --format json
```

#### Validate Dataset
```bash
# Basic validation
python dataset_cli.py validate --id dataset_123

# With custom rules
python dataset_cli.py validate --id dataset_123 --rules validation_rules.json --output results.json

# Validate with quality metrics
python dataset_cli.py validate --id dataset_123 --quality-metrics
```

#### Process Dataset
```bash
# Filter data
python dataset_cli.py process --id dataset_123 --transformations '[{"operation": "filter", "params": {"condition": "age > 30"}}]'

# Multiple transformations
python dataset_cli.py process --id dataset_123 --transformations '[{"operation": "filter", "params": {"condition": "age > 25"}}, {"operation": "sort", "params": {"columns": ["salary"], "ascending": false}}]'

# Column operations
python dataset_cli.py process --id dataset_123 --transformations '[{"operation": "drop_columns", "params": {"columns": ["temp_col"]}}]'
```

#### Visualize Dataset
```bash
# Scatter plot
python dataset_cli.py visualize --id dataset_123 --type scatter --x age --y salary --save

# Histogram
python dataset_cli.py visualize --id dataset_123 --type histogram --column age --save

# Correlation matrix
python dataset_cli.py visualize --id dataset_123 --type correlation --save
```

#### Export Dataset
```bash
# Export to JSON
python dataset_cli.py export --id dataset_123 --format json --output data.json

# Export to Parquet
python dataset_cli.py export --id dataset_123 --format parquet --output data.parquet

# Export to Excel
python dataset_cli.py export --id dataset_123 --format excel --output data.xlsx
```

#### Trust Scoring
```bash
# Calculate trust score
python dataset_cli.py trust-score --id dataset_123 --method ensemble

# Quality assessment
python dataset_cli.py quality-assessment --id dataset_123

# Cleanlab comparison
python dataset_cli.py cleanlab-compare --id dataset_123 --option 1 --method side-by-side
```

#### List and Manage
```bash
# List all datasets
python dataset_cli.py list

# List in JSON format
python dataset_cli.py list --format json

# Delete dataset
python dataset_cli.py delete --id dataset_123 --force

# Get dataset info
python dataset_cli.py info --id dataset_123
```

### Advanced Commands

#### Batch Operations
```bash
# Batch trust scoring
python dataset_cli.py batch-trust-score --input-dir ./datasets --output results.json

# Batch validation
python dataset_cli.py batch-validate --input-dir ./datasets --output validation_results.json

# Batch export
python dataset_cli.py batch-export --input-dir ./datasets --format parquet --output-dir ./exports
```

#### Quality-Based Filtering
```bash
# Filter by trust score
python dataset_cli.py quality-filter --id dataset_123 --min-trust 0.8

# Filter with specific features
python dataset_cli.py quality-filter --id dataset_123 --min-trust 0.8 --features age,salary

# Filter with custom quality rules
python dataset_cli.py quality-filter --id dataset_123 --quality-rules rules.json
```

#### Report Generation
```bash
# Generate quality report
python dataset_cli.py quality-report --id dataset_123 --output report.json

# Generate trust score report
python dataset_cli.py trust-report --id dataset_123 --output trust_report.json

# Generate comparison report
python dataset_cli.py comparison-report --id dataset_123 --cleanlab-option 1 --output comparison.json
```

## 🔧 API Reference

### Core Classes

#### TrustScoringDashboard
Main dashboard class for the Streamlit application.

```python
class TrustScoringDashboard:
    def __init__(self, db_path: str = "./trust_scoring_dashboard.db")
    
    # File upload methods
    def upload_file_from_streamlit(self, uploaded_file) -> Optional[str]
    def upload_from_s3(self, bucket_name: str, file_key: str, ...) -> Optional[str]
    def upload_from_google_drive(self, file_id: str, ...) -> Optional[str]
    def upload_from_google_cloud_storage(self, bucket_name: str, blob_name: str, ...) -> Optional[str]
    def upload_from_local_path(self, file_path: str) -> Optional[str]
    
    # Data processing methods
    def load_data_by_format(self, dataset_path: str, data_format: str) -> Optional[pd.DataFrame]
    def validate_file_format(self, file_path: str) -> Tuple[bool, str]
    def get_supported_formats(self) -> List[str]
    
    # Trust scoring methods
    def execute_trust_scoring_command(self, command: str, dataset_path: str = None) -> Dict[str, Any]
    def run_cleanlab_comparison(self, dataset_path: str, data_format: str, ...) -> Dict[str, Any]
    
    # Dashboard methods
    def run_streamlit_dashboard(self)
    def show_overview_page(self, data: Dict[str, Any])
    def show_trust_scoring_page(self, data: Dict[str, Any])
    def show_analytics_page(self, data: Dict[str, Any])
    def show_commands_page(self, data: Dict[str, Any])
    def show_testing_page(self, data: Dict[str, Any])
    def show_reports_page(self, data: Dict[str, Any])
    def show_sql_query_page(self, data: Dict[str, Any])
```

#### AdvancedTrustScoringEngine
Advanced trust scoring engine with multiple methods.

```python
class AdvancedTrustScoringEngine:
    def __init__(self)
    
    def calculate_advanced_trust_score(self, dataset: pd.DataFrame) -> Dict[str, Any]
    def calculate_ensemble_trust_score(self, dataset: pd.DataFrame) -> float
    def calculate_robust_trust_score(self, dataset: pd.DataFrame) -> float
    def calculate_uncertainty_trust_score(self, dataset: pd.DataFrame) -> float
```

#### FallbackDataQualityManager
Data quality manager with fallback capabilities.

```python
class FallbackDataQualityManager:
    def __init__(self)
    
    def calculate_data_trust_score(self, dataset: pd.DataFrame) -> Dict[str, Any]
    def create_quality_based_filter(self, dataset: pd.DataFrame, min_trust_score: float = 0.8) -> pd.DataFrame
    def generate_quality_report(self, dataset: pd.DataFrame) -> Dict[str, Any]
```

### Database Schema

#### Tables
```sql
-- Trust scores table
CREATE TABLE trust_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_name TEXT,
    dataset_id TEXT,
    trust_score REAL,
    method TEXT,
    component_scores TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    session_id TEXT
);

-- Quality metrics table
CREATE TABLE quality_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_name TEXT,
    dataset_id TEXT,
    missing_values_ratio REAL,
    duplicate_rows_ratio REAL,
    outlier_ratio REAL,
    data_completeness REAL,
    data_consistency REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    session_id TEXT
);

-- Test results table
CREATE TABLE test_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_name TEXT,
    test_type TEXT,
    status TEXT,
    duration REAL,
    details TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    session_id TEXT
);

-- System metrics table
CREATE TABLE system_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name TEXT,
    metric_value REAL,
    metric_unit TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    session_id TEXT
);

-- Commands executed table
CREATE TABLE commands_executed (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    command TEXT,
    status TEXT,
    output TEXT,
    duration REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    session_id TEXT
);
```

## 📚 Examples

### Example 1: Basic Trust Scoring
```python
from data_engineering.trust_scoring_dashboard import TrustScoringDashboard
import pandas as pd

# Initialize dashboard
dashboard = TrustScoringDashboard()

# Create sample data
data = {
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [10, 20, 30, 40, 50],
    'feature3': [100, 200, 300, 400, 500]
}
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('sample_data.csv', index=False)

# Calculate trust score
result = dashboard.execute_trust_scoring_command("calculate_trust_score", "sample_data.csv")
print(f"Trust Score: {result.get('trust_score', 'N/A'):.3f}")
```

### Example 2: File Upload and Processing
```python
from data_engineering.trust_scoring_dashboard import TrustScoringDashboard

# Initialize dashboard
dashboard = TrustScoringDashboard()

# Upload from local path
uploaded_path = dashboard.upload_from_local_path("path/to/your/data.csv")
if uploaded_path:
    print(f"File uploaded: {uploaded_path}")
    
    # Validate format
    is_valid, format_type = dashboard.validate_file_format(uploaded_path)
    print(f"Format: {format_type}, Valid: {is_valid}")
    
    # Load data
    df = dashboard.load_data_by_format(uploaded_path, format_type)
    print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
```

### Example 3: Cleanlab Comparison
```python
from data_engineering.trust_scoring_dashboard import TrustScoringDashboard

# Initialize dashboard
dashboard = TrustScoringDashboard()

# Run Cleanlab comparison
result = dashboard.run_cleanlab_comparison(
    dataset_path="path/to/data.csv",
    data_format="CSV",
    cleanlab_option=1,
    comparison_method="Side-by-Side",
    visualization_type="Both"
)

if "error" not in result:
    print(f"Our Score: {result['our_score']:.3f}")
    print(f"Cleanlab Score: {result['cleanlab_score']:.3f}")
    print(f"Analysis: {result['comparison_analysis']}")
```

### Example 4: Custom Validation Rules
```python
from data_engineering.trust_scoring_dashboard import TrustScoringDashboard

# Initialize dashboard
dashboard = TrustScoringDashboard()

# Define custom validation rules
validation_rules = {
    "age_range": lambda df: (df['age'] >= 0) & (df['age'] <= 120),
    "salary_positive": lambda df: df['salary'] > 0,
    "unique_ids": lambda df: df['id'].is_unique
}

# Apply validation
# (Implementation depends on specific validation method)
```

### Example 5: Batch Processing
```python
import os
from data_engineering.trust_scoring_dashboard import TrustScoringDashboard

# Initialize dashboard
dashboard = TrustScoringDashboard()

# Process multiple files
data_dir = "./datasets"
results = []

for file_path in os.listdir(data_dir):
    if file_path.endswith('.csv'):
        full_path = os.path.join(data_dir, file_path)
        
        # Calculate trust score
        result = dashboard.execute_trust_scoring_command("calculate_trust_score", full_path)
        results.append({
            'file': file_path,
            'trust_score': result.get('trust_score', 0),
            'method': result.get('method', 'unknown')
        })

# Display results
for result in results:
    print(f"{result['file']}: {result['trust_score']:.3f} ({result['method']})")
```

## 🚨 Troubleshooting

### Common Issues

#### 1. File Upload Issues
**Problem**: File upload fails
```bash
# Solution
- Check file format is supported
- Verify file size (max 200MB)
- Ensure file is not corrupted
- Check file permissions
```

#### 2. Cloud Storage Issues
**Problem**: S3/Google Drive upload fails
```bash
# Solution
- Verify credentials are correct
- Check network connectivity
- Ensure bucket/file exists
- Verify permissions
```

#### 3. Trust Scoring Issues
**Problem**: Trust scoring fails
```bash
# Solution
- Check data format is valid
- Ensure sufficient data (min 10 rows)
- Verify numeric columns exist
- Check for missing values
```

#### 4. Dashboard Issues
**Problem**: Dashboard won't start
```bash
# Solution
- Verify all dependencies installed
- Check port availability
- Ensure virtual environment activated
- Check Python version (3.8+)
```

#### 5. Database Issues
**Problem**: Database errors
```bash
# Solution
- Check database file permissions
- Verify SQLite is available
- Check disk space
- Restart application
```

### Error Messages

#### File Format Errors
```
Error: Unsupported file format: .txt
Solution: Use supported formats (CSV, JSON, Excel, etc.)
```

#### Cloud Storage Errors
```
Error: AWS credentials not found
Solution: Provide AWS access key and secret key
```

#### Trust Scoring Errors
```
Error: Insufficient data for trust scoring
Solution: Ensure dataset has at least 10 rows and numeric columns
```

#### Database Errors
```
Error: Database locked
Solution: Close other applications using the database
```

### Performance Optimization

#### Large Files
```python
# For files > 100MB
- Use chunked processing
- Enable streaming upload
- Use efficient formats (Parquet, HDF5)
- Consider data sampling
```

#### Multiple Users
```python
# For concurrent users
- Use separate database files
- Implement session management
- Enable caching
- Use load balancing
```

#### Memory Issues
```python
# For memory constraints
- Use data streaming
- Implement garbage collection
- Use efficient data structures
- Consider data sampling
```

## 📞 Support

### Getting Help
- **Documentation**: Check this README first
- **Issues**: Report issues on GitHub
- **Discussions**: Use GitHub Discussions
- **Email**: Contact maintainers directly

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### License
This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🎯 Quick Reference

### Dashboard URLs
- **Local**: http://localhost:8501
- **Network**: http://your-ip:8501

### Key Commands
```bash
# Start dashboard
streamlit run trust_scoring_dashboard.py

# Test upload functionality
python test_file_upload.py

# Create test files
python create_test_files.py

# CLI operations
python dataset_cli.py --help
```

### Supported Formats
- **Data**: CSV, JSON, Excel, Parquet, HDF5
- **Advanced**: Pickle, Feather, Stata, SAS, SPSS, XML, YAML

### Upload Sources
- **Local**: Drag & drop files
- **S3**: Amazon S3 buckets
- **Google Drive**: Google Drive files
- **GCS**: Google Cloud Storage
- **Local Path**: File system paths

### Trust Scoring Methods
- **Ensemble**: Robust, reliable scoring
- **Robust**: Outlier-resistant methods
- **Uncertainty**: Uncertainty quantification

### Cleanlab Options
- **Option 1**: Basic label quality
- **Option 2**: Confidence-based
- **Option 3**: Uncertainty-based
- **Option 4**: Ensemble methods 