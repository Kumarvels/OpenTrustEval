import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
End-to-End Data Engineering Pipeline Example Script

Demonstrates:
- Synthetic data creation
- Data upload
- DB load (Snowflake)
- ELT with Spark, dbt, Airflow
- DB tuning
- Data versioning
- Dashboard/report
- Integration test

Dependencies:
- Python 3.8+
- All connectors in data_engineering.connectors (stubs provided)
- No external services required for this stub/demo

To run:
$ python3 data_engineering/example_end_to_end_pipeline.py
"""
from data_engineering.data_lifecycle import (
    DataLifecycleManager, SparkConnector, DBTConnector, AirflowConnector, SnowflakeConnector, PowerBIConnector
)

def setup_manager():
    """Initialize and return a DataLifecycleManager with all required connectors and DBs."""
    manager = DataLifecycleManager()
    manager.add_connector('spark', SparkConnector({'master': 'local'}))
    manager.add_connector('dbt', DBTConnector('/my/dbt/project'))
    manager.add_connector('airflow', AirflowConnector('http://localhost:8080', 'token'))
    manager.add_connector('powerbi', PowerBIConnector('workspace', 'token'))
    manager.add_db('snowflake', SnowflakeConnector('account', 'user', 'pass', 'db'))
    return manager

def run_pipeline(manager):
    """Run the full data engineering pipeline and perform integration tests."""
    # Step 1: Synthetic data
    manager.generate_synthetic_data({'rows': 1000, 'schema': {'id': 'int', 'value': 'float'}})
    # Step 2: Upload data
    manager.upload_data('file', '/tmp/synthetic.csv')
    # Step 3: ELT pipeline
    elt_pipeline = [
        {'tool': 'spark', 'action': 'run_job', 'args': {'job': 'elt_job'}},
        {'tool': 'dbt', 'action': 'run', 'args': {'model': 'clean_model'}},
        {'tool': 'airflow', 'action': 'trigger_dag', 'args': {'dag_id': 'finalize_data'}},
    ]
    manager.run_pipeline(elt_pipeline)
    # Step 4: Tune DB
    manager.tune_db('snowflake', {'indexes': True, 'partitions': ['date']})
    # Step 5: Version data
    manager.version_data()
    # Step 6: Dashboard/report
    manager.dashboard('powerbi', {'title': 'Final Clean Data Report'})
    # Step 7: Integration test (assertions)
    assert manager.metrics['synthetic_data_generated'] == 1, 'Synthetic data step failed'
    assert manager.metrics['uploads'] == 1, 'Upload step failed'
    assert manager.metrics['elt_runs'] == 3, 'ELT pipeline step failed'
    assert manager.metrics['db_tuned'] == 1, 'DB tuning step failed'
    assert manager.metrics['versioned'] == 1, 'Versioning step failed'
    print('Integration test passed!')
    print('Governance logs:', manager.get_governance_logs())
    print('Metrics:', manager.get_metrics())

def main():
    print("--- Running comprehensive end-to-end pipeline example ---")
    manager = setup_manager()
    try:
        run_pipeline(manager)
    except AssertionError as e:
        print(f"Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
