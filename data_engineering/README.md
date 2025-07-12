# Data Engineering with Dataset Management

This section provides comprehensive data engineering capabilities with integrated dataset management features for dataset creation, validation, processing, visualization, and export/import operations.

## 🚀 Quick Start

### Installation
```bash
pip install pandas numpy pyyaml
# Optional for advanced features
pip install plotly great-expectations pyarrow openpyxl
# Optional for Cleanlab benchmarking comparison only
pip install cleanlab scikit-learn
```

### Basic Usage
```python
from data_engineering.data_lifecycle import DataLifecycleManager

# Initialize manager with dataset management support
manager = DataLifecycleManager()

# Create a dataset
sample_data = {'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']}
dataset_id = manager.create_dataset('employees', sample_data)

# Validate dataset
results = manager.validate_dataset(dataset_id)
print(f"Validation: {'PASSED' if results['passed'] else 'FAILED'}")

# Process dataset
transformations = [{'operation': 'filter', 'params': {'condition': 'id > 1'}}]
new_dataset_id = manager.process_dataset(dataset_id, transformations)

# Visualize dataset
viz_config = {'type': 'scatter', 'x': 'id', 'y': 'name', 'save': True}
viz_path = manager.visualize_dataset(dataset_id, viz_config)
```

## 📁 Dataset Management Features

### 1. Dataset Management
- **Create**: Create datasets from various sources (CSV, JSON, Parquet, Excel)
- **Load**: Load datasets by ID with metadata
- **List**: List all available datasets with statistics
- **Delete**: Remove datasets with confirmation

### 2. Data Validation
- **Schema Validation**: Validate against defined schemas
- **Type Checking**: Ensure data types match expectations
- **Null Detection**: Identify missing values
- **Custom Rules**: Apply custom validation functions
- **Quality Metrics**: Generate data quality statistics

### 3. Data Processing
- **Filtering**: Filter rows based on conditions
- **Sorting**: Sort data by columns
- **Column Operations**: Drop, rename, and transform columns
- **Aggregations**: Group and aggregate data
- **Custom Transformations**: Apply custom processing functions

### 4. Data Visualization
- **Histograms**: Distribution analysis
- **Scatter Plots**: Correlation analysis
- **Bar Charts**: Categorical data visualization
- **Line Charts**: Time series analysis
- **Correlation Matrices**: Numeric data relationships
- **Export**: Save visualizations as HTML files

### 5. Export/Import
- **Multiple Formats**: CSV, JSON, Parquet, Excel
- **Auto-detection**: Automatic format detection
- **Custom Paths**: Specify output locations
- **Batch Operations**: Process multiple datasets

## 🎯 **Advanced Quality Assessment (Fallback System)**

### **Quality Assessment Features**
The system includes advanced data quality assessment with **automatic fallback support**:

#### **Core System (Always Available)**
- **Statistical Quality Metrics**: Missing values, duplicates, outliers
- **Trust Scoring**: Advanced statistical trust assessment
- **Quality Filtering**: Row-wise quality filtering
- **Automated Validation**: Statistical validation rules

#### **Cleanlab Benchmarking (Optional)**
- **Trust Score Comparison**: Compare our trust scores against Cleanlab's
- **Validation**: Validate our advanced methods against industry standard
- **Benchmarking**: Performance comparison and analysis

### **Quality Assessment Methods**
```python
from data_engineering.cleanlab_integration import FallbackDataQualityManager

# Initialize fallback quality manager (always works)
manager = FallbackDataQualityManager()

# Calculate trust score
trust_result = manager.calculate_data_trust_score(dataset)
print(f"Trust Score: {trust_result['trust_score']:.3f}")
print(f"Method: {trust_result['method']}")  # 'fallback_statistical'

# Quality-based filtering
filtered_data = manager.create_quality_based_filter(dataset, min_trust_score=0.8)

# Generate quality report
report = manager.generate_quality_report(dataset)

# Optional: Benchmark against Cleanlab (if installed)
from data_engineering.cleanlab_integration import benchmark_vs_cleanlab
benchmark_results = benchmark_vs_cleanlab(dataset, labels)
print(f"Our Score: {benchmark_results['our_trust_score']:.3f}")
print(f"Cleanlab Score: {benchmark_results['cleanlab_trust_score']:.3f}")
```

## 🛠️ Command Line Interface

### Installation
The CLI is available at `data_engineering/scripts/dataset_cli.py`

### Usage Examples

#### Create Dataset
```bash
# Create from CSV
python dataset_cli.py create --name "my_dataset" --input data.csv --format csv

# Create with schema
python dataset_cli.py create --name "employees" --input employees.csv --format csv --schema schema.json
```

#### Validate Dataset
```bash
# Basic validation
python dataset_cli.py validate --id dataset_123

# With custom rules
python dataset_cli.py validate --id dataset_123 --rules validation_rules.json --output results.json
```

#### Process Dataset
```bash
# Filter data
python dataset_cli.py process --id dataset_123 --transformations '[{"operation": "filter", "params": {"condition": "age > 30"}}]'

# Multiple transformations
python dataset_cli.py process --id dataset_123 --transformations '[{"operation": "filter", "params": {"condition": "age > 25"}}, {"operation": "sort", "params": {"columns": ["salary"], "ascending": false}}]'
```

#### Visualize Dataset
```bash
# Scatter plot
python dataset_cli.py visualize --id dataset_123 --type scatter --x age --y salary --save

# Histogram
python dataset_cli.py visualize --id dataset_123 --type histogram --column age --save
```

#### Export Dataset
```bash
# Export to JSON
python dataset_cli.py export --id dataset_123 --format json --output data.json

# Export to Parquet
python dataset_cli.py export --id dataset_123 --format parquet --output data.parquet
```

#### List and Manage
```bash
# List all datasets
python dataset_cli.py list

# List in JSON format
python dataset_cli.py list --format json

# Delete dataset
python dataset_cli.py delete --id dataset_123 --force
```

#### Quality-Based Filtering (Works with/without Cleanlab)
```bash
# Filter by trust score (uses fallback if Cleanlab unavailable)
python dataset_cli.py quality-filter --id dataset_123 --min-trust 0.8

# Optionally specify features
python dataset_cli.py quality-filter --id dataset_123 --min-trust 0.8 --features age,salary
```

#### Generate Quality Report (Works with/without Cleanlab)
```bash
# Generate quality report (uses fallback if Cleanlab unavailable)
python dataset_cli.py quality-report --id dataset_123

# Save to file
python dataset_cli.py quality-report --id dataset_123 --output report.json
```

## 🌐 Web User Interface

### Launch WebUI
```bash
python data_engineering/scripts/easy_dataset_webui.py
```

The WebUI will be available at `http://localhost:7861`

### WebUI Features
- **Create Dataset**: Upload files and create datasets with schema/metadata
- **Dataset Management**: List, load, and manage all datasets
- **Validation**: Validate datasets with custom rules
- **Processing**: Apply transformations through a user-friendly interface
- **Visualization**: Create charts and graphs with interactive controls
- **Export/Import**: Export datasets to various formats
- **Delete**: Remove datasets with confirmation
- **Quality-Based Filtering (Cleanlab)**: Filter datasets by trust score (new tab) - **Works with fallback system**
- **Quality Report (Cleanlab)**: Generate and download a comprehensive quality report (new tab) - **Works with fallback system**

## 🔧 Advanced Usage

### Custom Validation Rules
```python
# Define custom validation functions
validation_rules = {
    "age_range": lambda df: (df['age'] >= 0) & (df['age'] <= 120),
    "salary_positive": lambda df: df['salary'] > 0,
    "unique_ids": lambda df: df['id'].is_unique
}

results = manager.validate_dataset(dataset_id, validation_rules)
```

### Complex Transformations
```python
# Multiple transformations
transformations = [
    {'operation': 'filter', 'params': {'condition': 'age > 25'}},
    {'operation': 'sort', 'params': {'columns': ['salary'], 'ascending': False}},
    {'operation': 'drop_columns', 'params': {'columns': ['temp_column']}},
    {'operation': 'rename_columns', 'params': {'mapping': {'old_name': 'new_name'}}}
]

new_dataset_id = manager.process_dataset(dataset_id, transformations)
```

### Quality Assessment Integration
```python
# Use quality assessment in data engineering pipeline
manager = DataLifecycleManager()

# Create dataset from external source
dataset_id = manager.import_dataset('external_data.csv', 'external_data')

# Assess data quality (works with/without Cleanlab)
if hasattr(manager.dataset_manager, 'cleanlab_manager'):
    trust_result = manager.dataset_manager.cleanlab_manager.calculate_data_trust_score(
        manager.load_dataset(dataset_id)
    )
    print(f"Data Trust Score: {trust_result['trust_score']:.3f}")
    print(f"Assessment Method: {trust_result['method']}")

# Validate data quality
validation_results = manager.validate_dataset(dataset_id)
if not validation_results['passed']:
    print("Data quality issues found:", validation_results['errors'])

# Process data with quality filtering
if hasattr(manager.dataset_manager, 'cleanlab_manager'):
    filtered_dataset_id = manager.dataset_manager.create_quality_filtered_dataset(
        dataset_id, min_trust_score=0.8
    )
    print(f"Quality-filtered dataset: {filtered_dataset_id}")

# Export for downstream processing
export_path = manager.export_dataset(dataset_id, 'parquet', 'processed_data.parquet')

# Create visualization for monitoring
viz_config = {
    'type': 'line',
    'x': 'timestamp',
    'y': 'value',
    'save': True
}
viz_path = manager.visualize_dataset(dataset_id, viz_config)
```

## 📊 Metrics and Monitoring

The DataLifecycleManager tracks various metrics:
- `datasets_created`: Number of datasets created
- `datasets_loaded`: Number of datasets loaded
- `datasets_validated`: Number of validation runs
- `datasets_processed`: Number of processing operations
- `visualizations_created`: Number of visualizations generated
- `datasets_exported`: Number of export operations
- `datasets_imported`: Number of import operations
- `datasets_deleted`: Number of deletions

```python
# Get metrics
metrics = manager.get_metrics()
print("Dataset operations:", metrics)

# Get governance logs
logs = manager.get_governance_logs()
print("Activity logs:", logs)
```

## 🔌 Connector Integration

Dataset management integrates with existing data engineering connectors:

```python
# Use with Spark
manager.add_connector('spark', SparkConnector({'master': 'local'}))
spark_df = manager.connectors['spark'].read_csv('data.csv')
dataset_id = manager.create_dataset('spark_data', spark_df.toPandas())

# Use with dbt
manager.add_connector('dbt', DBTConnector('/my/dbt/project'))
manager.connectors['dbt'].run(model='my_model')

# Use with Snowflake
manager.add_db('snowflake', SnowflakeConnector('account', 'user', 'pass', 'db'))
snowflake_data = manager.sql_transform('snowflake', 'SELECT * FROM my_table')
dataset_id = manager.create_dataset('snowflake_data', snowflake_data)
```

## 📝 Configuration

### Schema Definition
```json
{
  "id": {
    "type": "int64",
    "nullable": false,
    "unique": true
  },
  "name": {
    "type": "string",
    "nullable": false
  },
  "age": {
    "type": "int64",
    "nullable": true
  },
  "salary": {
    "type": "float64",
    "nullable": true
  }
}
```

### Validation Rules
```json
{
  "age_range": "lambda df: (df['age'] >= 0) & (df['age'] <= 120)",
  "salary_positive": "lambda df: df['salary'] > 0",
  "unique_ids": "lambda df: df['id'].is_unique"
}
```

### Transformation Examples
```json
[
  {
    "operation": "filter",
    "params": {
      "condition": "age > 25"
    }
  },
  {
    "operation": "sort",
    "params": {
      "columns": ["salary"],
      "ascending": false
    }
  },
  {
    "operation": "drop_columns",
    "params": {
      "columns": ["temp_column"]
    }
  }
]
```

## 🚀 Best Practices

1. **Always validate data** before processing
2. **Use schemas** to ensure data consistency
3. **Version your datasets** for reproducibility
4. **Monitor data quality** with regular validation
5. **Document transformations** for audit trails
6. **Use appropriate formats** for different use cases
7. **Backup important datasets** before major operations
8. **Leverage quality assessment** for data filtering and validation
9. **Use fallback systems** when advanced dependencies are unavailable

## 🔗 Related Documentation

- [Data Engineering Lifecycle](data_lifecycle.py)
- [Common Data Loader](common_data_loader.py)
- [Cleanlab Integration](cleanlab_integration.py) - **New: Quality assessment with fallback support**
- [Example Projects](project_*/)
- [Connectors](connectors/) 