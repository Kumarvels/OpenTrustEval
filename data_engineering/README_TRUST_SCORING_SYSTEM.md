# 🔍 Trust Scoring System - Batch Testing & Dashboard

A comprehensive system for testing, monitoring, and analyzing trust scoring capabilities with batch processing and real-time dashboard visualization.

## 🚀 Quick Start

### Installation

```bash
# Install dashboard dependencies
pip install -r requirements_dashboard.txt

# Optional: Install advanced features
pip install scikit-learn cleanlab
```

### Run the System

```bash
# Interactive launcher
python data_engineering/run_trust_scoring_system.py

# Or use specific commands:
python data_engineering/run_trust_scoring_system.py --test        # Quick test
python data_engineering/run_trust_scoring_system.py --batch       # Batch test suite
python data_engineering/run_trust_scoring_system.py --dashboard   # Start dashboard
python data_engineering/run_trust_scoring_system.py --all         # Run everything
```

## 📋 System Components

### 1. Batch Test Suite (`batch_trust_scoring_test_suite.py`)

**Comprehensive testing framework that executes all trust scoring commands in correct order:**

#### Features:
- **10-Phase Testing**: Complete system validation
- **Component Initialization**: Verify all components load correctly
- **Synthetic Data Generation**: Create test datasets with various quality levels
- **Advanced Trust Scoring**: Test ensemble, robust, and uncertainty methods
- **Fallback Quality Manager**: Test quality assessment capabilities
- **Dataset Integration**: Test dataset management features
- **Real-World Trust Scoring**: Test with realistic datasets
- **Cleanlab Benchmarking**: Compare against industry standards
- **Performance & Stress Testing**: Validate system under load
- **Integration & E2E Testing**: Complete pipeline validation

#### Usage:
```python
from data_engineering.batch_trust_scoring_test_suite import BatchTrustScoringTestSuite

# Run complete test suite
test_suite = BatchTrustScoringTestSuite()
results = test_suite.run_complete_test_suite()

# Check results
print(f"Tests completed: {len(results)}")
print(f"Success rate: {results.get('success_rate', 'N/A')}")
```

#### Test Phases:
1. **Component Initialization**: Verify all components load
2. **Synthetic Data Generation**: Create test datasets
3. **Advanced Trust Scoring**: Test scoring methods
4. **Fallback Quality Manager**: Test quality assessment
5. **Dataset Integration**: Test dataset management
6. **Real-World Trust Scoring**: Test with real datasets
7. **Cleanlab Benchmarking**: Compare with industry standards
8. **Performance & Stress Testing**: Load testing
9. **Integration & E2E Testing**: Complete pipeline
10. **Report Generation**: Comprehensive reporting

### 2. Trust Scoring Dashboard (`trust_scoring_dashboard.py`)

**Lightweight data science dashboard with SQL support for 360° monitoring:**

#### Features:
- **📊 Overview**: System health and key metrics
- **🔍 Trust Scoring**: Real-time trust score calculation and monitoring
- **📈 Analytics**: Advanced analytics and insights
- **⚡ Commands**: Command execution and monitoring
- **🧪 Testing**: Test execution and results
- **📋 Reports**: Report generation and export
- **🗄️ SQL Query**: Direct SQL query interface

#### Dashboard Pages:

##### 📊 Overview Page
- Key metrics dashboard
- Recent activity monitoring
- System health indicators
- Real-time visualizations

##### 🔍 Trust Scoring Page
- Calculate trust scores for datasets
- Quality assessment tools
- Historical trust score analysis
- Method comparison

##### 📈 Analytics Page
- Quality metrics analysis
- Performance analytics
- Correlation analysis
- Trend visualization

##### ⚡ Commands Page
- Execute trust scoring commands
- Monitor command performance
- View command history
- Real-time execution status

##### 🧪 Testing Page
- Run various test types
- Monitor test results
- Performance benchmarking
- Test result analysis

##### 📋 Reports Page
- Generate comprehensive reports
- Export analytics
- System metrics monitoring
- Custom report creation

##### 🗄️ SQL Query Page
- Direct SQL query interface
- Example queries provided
- Real-time data exploration
- Custom analytics

#### Usage:
```bash
# Start dashboard
streamlit run data_engineering/trust_scoring_dashboard.py

# Or use launcher
python data_engineering/run_trust_scoring_system.py --dashboard
```

### 3. System Launcher (`run_trust_scoring_system.py`)

**Easy-to-use launcher for all system components:**

#### Features:
- Dependency checking
- Interactive mode
- Command-line interface
- Error handling
- Component orchestration

#### Usage Examples:
```bash
# Interactive mode
python data_engineering/run_trust_scoring_system.py

# Quick test
python data_engineering/run_trust_scoring_system.py --test

# Batch testing
python data_engineering/run_trust_scoring_system.py --batch

# Dashboard only
python data_engineering/run_trust_scoring_system.py --dashboard --port 8502

# Run everything
python data_engineering/run_trust_scoring_system.py --all
```

## 🗄️ Database Schema

The dashboard uses SQLite with the following schema:

### Tables:

#### `trust_scores`
- `id`: Primary key
- `dataset_name`: Name of the dataset
- `dataset_id`: Unique dataset identifier
- `trust_score`: Calculated trust score
- `method`: Scoring method used
- `component_scores`: JSON of component scores
- `timestamp`: When score was calculated
- `session_id`: Session identifier

#### `quality_metrics`
- `id`: Primary key
- `dataset_name`: Name of the dataset
- `dataset_id`: Unique dataset identifier
- `missing_values_ratio`: Ratio of missing values
- `duplicate_rows_ratio`: Ratio of duplicate rows
- `outlier_ratio`: Ratio of outliers
- `data_completeness`: Data completeness score
- `data_consistency`: Data consistency score
- `timestamp`: When metrics were calculated
- `session_id`: Session identifier

#### `test_results`
- `id`: Primary key
- `test_name`: Name of the test
- `test_type`: Type of test
- `status`: Test status (completed/failed)
- `duration`: Test duration in seconds
- `details`: JSON with test details
- `timestamp`: When test was run
- `session_id`: Session identifier

#### `system_metrics`
- `id`: Primary key
- `metric_name`: Name of the metric
- `metric_value`: Metric value
- `metric_unit`: Unit of measurement
- `timestamp`: When metric was recorded
- `session_id`: Session identifier

#### `commands_executed`
- `id`: Primary key
- `command`: Command executed
- `status`: Execution status
- `output`: Command output
- `duration`: Execution duration
- `timestamp`: When command was executed
- `session_id`: Session identifier

## 📊 Example SQL Queries

### Recent Trust Scores
```sql
SELECT dataset_name, trust_score, method, timestamp 
FROM trust_scores 
ORDER BY timestamp DESC 
LIMIT 10;
```

### Quality Issues Analysis
```sql
SELECT dataset_name, missing_values_ratio, duplicate_rows_ratio, outlier_ratio 
FROM quality_metrics 
ORDER BY timestamp DESC 
LIMIT 10;
```

### Test Success Rate
```sql
SELECT test_type, 
       COUNT(*) as total_tests, 
       SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_tests 
FROM test_results 
GROUP BY test_type;
```

### Command Performance
```sql
SELECT command, 
       AVG(duration) as avg_duration, 
       COUNT(*) as execution_count 
FROM commands_executed 
GROUP BY command 
ORDER BY avg_duration DESC;
```

### System Health
```sql
SELECT metric_name, 
       AVG(metric_value) as avg_value, 
       MAX(metric_value) as max_value 
FROM system_metrics 
GROUP BY metric_name;
```

## 🔧 Configuration

### Dashboard Configuration
The dashboard can be configured through environment variables:

```bash
# Dashboard port
export STREAMLIT_SERVER_PORT=8501

# Dashboard address
export STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Database path
export TRUST_SCORING_DB_PATH=./trust_scoring_dashboard.db
```

### Test Suite Configuration
The batch test suite can be configured:

```python
# Custom output directory
test_suite = BatchTrustScoringTestSuite(output_dir="./custom_reports")

# Run specific phases
test_suite._test_component_initialization()
test_suite._test_advanced_trust_scoring()
```

## 📈 Monitoring & Analytics

### Key Metrics Tracked:
- **Trust Scores**: Historical trust score trends
- **Quality Metrics**: Data quality over time
- **Performance**: System performance metrics
- **Test Results**: Test success rates and durations
- **Command Execution**: Command performance and success rates

### Visualizations:
- **Line Charts**: Time series analysis
- **Scatter Plots**: Correlation analysis
- **Histograms**: Distribution analysis
- **Box Plots**: Performance analysis
- **Pie Charts**: Summary statistics

### Real-time Monitoring:
- Live dashboard updates
- Real-time metrics collection
- Instant command execution
- Live test monitoring

## 🚀 Advanced Usage

### Custom Test Datasets
```python
# Create custom test dataset
custom_data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 1000),
    'feature2': np.random.normal(0, 1, 1000),
    'feature3': np.random.normal(0, 1, 1000)
})

# Test with custom data
result = test_suite.advanced_engine.calculate_advanced_trust_score(custom_data)
```

### Custom SQL Queries
```python
# Execute custom SQL
dashboard = TrustScoringDashboard()
result = dashboard.execute_sql_query("""
    SELECT AVG(trust_score) as avg_score, 
           COUNT(*) as total_scores 
    FROM trust_scores 
    WHERE method = 'ensemble'
""")
```

### Custom Reports
```python
# Generate custom report
report = dashboard.generate_report("custom_report", data)
```

## 🔍 Troubleshooting

### Common Issues:

1. **Import Errors**: Install missing dependencies
   ```bash
   pip install -r requirements_dashboard.txt
   ```

2. **Port Already in Use**: Use different port
   ```bash
   python run_trust_scoring_system.py --dashboard --port 8502
   ```

3. **Database Errors**: Check file permissions
   ```bash
   chmod 755 ./trust_scoring_dashboard.db
   ```

4. **Memory Issues**: Reduce dataset size for testing

### Debug Mode:
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 API Reference

### BatchTrustScoringTestSuite
- `run_complete_test_suite()`: Run all tests
- `_test_component_initialization()`: Test component loading
- `_test_advanced_trust_scoring()`: Test trust scoring
- `_test_fallback_quality_manager()`: Test quality assessment
- `_test_dataset_integration()`: Test dataset management
- `_test_real_world_trust_scoring()`: Test with real data
- `_test_cleanlab_benchmarking()`: Test benchmarking
- `_test_performance_and_stress()`: Performance testing
- `_test_integration_and_e2e()`: End-to-end testing
- `_generate_comprehensive_report()`: Generate reports

### TrustScoringDashboard
- `run_streamlit_dashboard()`: Start dashboard
- `execute_sql_query(query)`: Execute SQL
- `log_trust_score(...)`: Log trust score
- `log_quality_metrics(...)`: Log quality metrics
- `log_test_result(...)`: Log test result
- `log_system_metric(...)`: Log system metric
- `log_command(...)`: Log command execution
- `generate_report(...)`: Generate reports

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the troubleshooting section
- Review the API reference
- Open an issue on GitHub
- Check the documentation

---

**🔍 OpenTrustEval - Trust Scoring System**  
*Comprehensive data quality and trust assessment platform* 