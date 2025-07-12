"""
Data Engineering Lifecycle Management (Segmented)
- Modular connectors for each tool/platform
- Orchestrate single-tool or multi-tool pipelines
- Python/SQL transform hooks
- Dashboard and suggestion technique integration
"""

from data_engineering.connectors.spark_connector import SparkConnector
from data_engineering.connectors.kafka_connector import KafkaConnector
from data_engineering.connectors.snowflake_connector import SnowflakeConnector
from data_engineering.connectors.airflow_connector import AirflowConnector
from data_engineering.connectors.dbt_connector import DBTConnector
from data_engineering.connectors.redshift_connector import RedshiftConnector
from data_engineering.connectors.bigquery_connector import BigQueryConnector
from data_engineering.connectors.powerbi_connector import PowerBIConnector
from data_engineering.connectors.hive_connector import HiveConnector
from data_engineering.connectors.looker_connector import LookerConnector
from data_engineering.connectors.mongodb_connector import MongoDBConnector
from data_engineering.connectors.postgresql_connector import PostgreSQLConnector
from data_engineering.connectors.fivetran_connector import FivetranConnector
from data_engineering.connectors.athena_connector import AthenaConnector
from data_engineering.connectors.databricks_connector import DatabricksConnector
from data_engineering.connectors.synapse_connector import SynapseConnector
from data_engineering.connectors.flink_connector import FlinkConnector
from data_engineering.connectors.hudi_connector import HudiConnector

# Import Easy Dataset integration
try:
    from data_engineering.easy_dataset_integration import EasyDatasetConnector
    EASY_DATASET_AVAILABLE = True
except ImportError:
    EASY_DATASET_AVAILABLE = False

class DataLifecycleManager:
    def __init__(self):
        self.metrics = {}
        self.governance_logs = []
        self.security_checks = []
        self.connectors = {}
        self.dbs = {}
        self.easy_dataset = None
        # ...initialize data sources, configs...
        
        # Initialize Easy Dataset if available
        if EASY_DATASET_AVAILABLE:
            self.easy_dataset = EasyDatasetConnector()
            self.add_connector('easy_dataset', self.easy_dataset)

    def add_connector(self, name, connector):
        self.connectors[name] = connector

    def remove_connector(self, name):
        if name in self.connectors:
            del self.connectors[name]

    def add_db(self, name, db):
        self.dbs[name] = db

    def remove_db(self, name):
        if name in self.dbs:
            del self.dbs[name]

    def run_pipeline(self, steps):
        """
        Orchestrate a pipeline of steps, each step is a dict:
        {'tool': 'spark', 'action': 'run_job', 'args': {...}}
        Automatically increments elt_runs if 'elt' or 'run_job' is in the action.
        """
        for step in steps:
            tool = step['tool']
            action = step['action']
            args = step.get('args', {})
            connector = self.connectors.get(tool)
            if connector and hasattr(connector, action):
                getattr(connector, action)(**args)
                self.log_governance(f"{tool}.{action} executed")
                # Increment elt_runs for ELT/ETL steps
                if action in ('elt', 'run_job', 'trigger_dag', 'run'):
                    self.metrics['elt_runs'] = self.metrics.get('elt_runs', 0) + 1

    def python_transform(self, data, fn):
        """Apply a Python function to data."""
        return fn(data)

    def sql_transform(self, db_name, query):
        """Run a SQL query on a registered DB connector."""
        db = self.dbs.get(db_name)
        if db and hasattr(db, 'execute_query'):
            return db.execute_query(query)

    def dashboard(self, tool, report):
        """Publish a dashboard/report using a BI tool connector."""
        connector = self.connectors.get(tool)
        if connector and hasattr(connector, 'publish_report'):
            connector.publish_report(report)

    def suggest_next_step(self, context):
        """Stub for suggestion techniques (ML, rules, etc)."""
        # ...suggestion logic...
        return "Try running dbt after Spark job."

    def generate_synthetic_data(self, spec):
        """Generate synthetic data based on a spec (stub for faker, SDV, etc.)."""
        # ...synthetic data logic...
        self.metrics['synthetic_data_generated'] = self.metrics.get('synthetic_data_generated', 0) + 1
        self.log_governance('Synthetic data generated')

    def upload_data(self, source_type, source):
        """Upload data from API, webhook, or file (stub)."""
        # ...upload logic...
        self.metrics['uploads'] = self.metrics.get('uploads', 0) + 1
        self.log_governance(f'Data uploaded from {source_type}')

    def elt(self, source, db_name, transform_fn=None):
        """ELT: Extract, Load, then Transform data into a DB (stub)."""
        # ...extract from source...
        # ...load to db...
        # ...transform in db (if transform_fn provided)...
        self.metrics['elt_runs'] = self.metrics.get('elt_runs', 0) + 1
        self.log_governance(f'ELT run to {db_name}')

    def tune_db(self, db_name, options):
        """Tune DB performance (stub for index, partition, etc.)."""
        # ...tuning logic...
        self.metrics['db_tuned'] = self.metrics.get('db_tuned', 0) + 1
        self.log_governance(f'DB tuned: {db_name}')

    def run_etl(self):
        """Run ETL pipeline and collect metrics."""
        # ...ETL logic...
        self.metrics['etl_runs'] = self.metrics.get('etl_runs', 0) + 1
        # ...collect more metrics...
        self.log_governance('ETL run')
        self.run_security_check('ETL')

    def validate_data(self):
        """Validate data and log results."""
        # ...validation logic...
        self.metrics['validation_checks'] = self.metrics.get('validation_checks', 0) + 1
        self.log_governance('Data validation')
        self.run_security_check('Validation')

    def version_data(self):
        """Version data for reproducibility."""
        # ...versioning logic...
        self.metrics['versioned'] = self.metrics.get('versioned', 0) + 1
        self.log_governance('Data versioning')

    def log_governance(self, action):
        """Log governance/audit actions for compliance."""
        self.governance_logs.append({'action': action})

    def run_security_check(self, context):
        """Perform security checks (access control, data privacy, etc)."""
        self.security_checks.append({'context': context, 'status': 'checked'})

    def get_metrics(self):
        return self.metrics

    def get_governance_logs(self):
        return self.governance_logs

    def get_security_checks(self):
        return self.security_checks

    def create_dataset(self, name: str, data, schema=None, metadata=None):
        """Create a new dataset using Easy Dataset"""
        if not self.easy_dataset:
            raise RuntimeError("Easy Dataset not available")
        
        dataset_id = self.easy_dataset.create_dataset(name, data, schema, metadata)
        self.metrics['datasets_created'] = self.metrics.get('datasets_created', 0) + 1
        self.log_governance(f'Dataset created: {name} ({dataset_id})')
        return dataset_id

    def load_dataset(self, dataset_id: str):
        """Load a dataset by ID"""
        if not self.easy_dataset:
            raise RuntimeError("Easy Dataset not available")
        
        df = self.easy_dataset.load_dataset(dataset_id)
        self.metrics['datasets_loaded'] = self.metrics.get('datasets_loaded', 0) + 1
        self.log_governance(f'Dataset loaded: {dataset_id}')
        return df

    def validate_dataset(self, dataset_id: str, validation_rules=None):
        """Validate a dataset"""
        if not self.easy_dataset:
            raise RuntimeError("Easy Dataset not available")
        
        results = self.easy_dataset.validate_dataset(dataset_id, validation_rules)
        self.metrics['datasets_validated'] = self.metrics.get('datasets_validated', 0) + 1
        self.log_governance(f'Dataset validated: {dataset_id} ({results["passed"]})')
        return results

    def process_dataset(self, dataset_id: str, transformations):
        """Process a dataset with transformations"""
        if not self.easy_dataset:
            raise RuntimeError("Easy Dataset not available")
        
        new_dataset_id = self.easy_dataset.process_dataset(dataset_id, transformations)
        self.metrics['datasets_processed'] = self.metrics.get('datasets_processed', 0) + 1
        self.log_governance(f'Dataset processed: {dataset_id} -> {new_dataset_id}')
        return new_dataset_id

    def visualize_dataset(self, dataset_id: str, visualization_config):
        """Create visualization for dataset"""
        if not self.easy_dataset:
            raise RuntimeError("Easy Dataset not available")
        
        viz_path = self.easy_dataset.visualize_dataset(dataset_id, visualization_config)
        self.metrics['visualizations_created'] = self.metrics.get('visualizations_created', 0) + 1
        self.log_governance(f'Visualization created: {dataset_id}')
        return viz_path

    def export_dataset(self, dataset_id: str, format_type, output_path=None):
        """Export dataset to various formats"""
        if not self.easy_dataset:
            raise RuntimeError("Easy Dataset not available")
        
        export_path = self.easy_dataset.export_dataset(dataset_id, format_type, output_path)
        self.metrics['datasets_exported'] = self.metrics.get('datasets_exported', 0) + 1
        self.log_governance(f'Dataset exported: {dataset_id} -> {export_path}')
        return export_path

    def import_dataset(self, file_path: str, name: str, format_type=None):
        """Import dataset from various formats"""
        if not self.easy_dataset:
            raise RuntimeError("Easy Dataset not available")
        
        dataset_id = self.easy_dataset.import_dataset(file_path, name, format_type)
        self.metrics['datasets_imported'] = self.metrics.get('datasets_imported', 0) + 1
        self.log_governance(f'Dataset imported: {name} ({dataset_id})')
        return dataset_id

    def list_datasets(self):
        """List all available datasets"""
        if not self.easy_dataset:
            raise RuntimeError("Easy Dataset not available")
        
        datasets = self.easy_dataset.list_datasets()
        self.log_governance(f'Datasets listed: {len(datasets)} found')
        return datasets

    def delete_dataset(self, dataset_id: str):
        """Delete a dataset"""
        if not self.easy_dataset:
            raise RuntimeError("Easy Dataset not available")
        
        success = self.easy_dataset.delete_dataset(dataset_id)
        if success:
            self.metrics['datasets_deleted'] = self.metrics.get('datasets_deleted', 0) + 1
            self.log_governance(f'Dataset deleted: {dataset_id}')
        return success

# Example: Single-tool and multi-tool pipeline
if __name__ == "__main__":
    manager = DataLifecycleManager()
    manager.add_connector('spark', SparkConnector({'master': 'local'}))
    manager.add_connector('dbt', DBTConnector('/my/dbt/project'))
    manager.add_db('snowflake', SnowflakeConnector('account', 'user', 'pass', 'db'))
    # Single-tool
    manager.connectors['spark'].run_job({'job': 'etl_job'})
    # Multi-tool pipeline
    pipeline = [
        {'tool': 'spark', 'action': 'run_job', 'args': {'job': 'etl_job'}},
        {'tool': 'dbt', 'action': 'run', 'args': {'model': 'my_model'}},
    ]
    manager.run_pipeline(pipeline)
    # Python transform
    data = [1, 2, 3]
    print(manager.python_transform(data, lambda d: [x * 2 for x in d]))
    # SQL transform
    manager.sql_transform('snowflake', 'SELECT * FROM my_table')
    # Dashboard
    manager.add_connector('powerbi', PowerBIConnector('workspace', 'token'))
    manager.dashboard('powerbi', {'title': 'My Report'})
    # Suggestion
    print(manager.suggest_next_step({'last_tool': 'spark'}))
    print(manager.get_governance_logs())

    # Comprehensive Example: End-to-End Data Engineering Pipeline
    # 1. Generate synthetic data
    # 2. Upload data (simulate API/file upload)
    # 3. Load to DB (e.g., Snowflake)
    # 4. ELT with Spark, dbt, Airflow
    # 5. Tune DB
    # 6. Version data
    # 7. Dashboard/report
    # 8. Integration test for pipeline

    def example_end_to_end_pipeline():
        manager = DataLifecycleManager()
        # Step 1: Synthetic data
        manager.generate_synthetic_data({'rows': 1000, 'schema': {'id': 'int', 'value': 'float'}})
        # Step 2: Upload data
        manager.upload_data('file', '/tmp/synthetic.csv')
        # Step 3: Load to DB
        manager.add_db('snowflake', SnowflakeConnector('account', 'user', 'pass', 'db'))
        # Step 4: ELT pipeline
        manager.add_connector('spark', SparkConnector({'master': 'local'}))
        manager.add_connector('dbt', DBTConnector('/my/dbt/project'))
        manager.add_connector('airflow', AirflowConnector('http://localhost:8080', 'token'))
        elt_pipeline = [
            {'tool': 'spark', 'action': 'run_job', 'args': {'job': 'elt_job'}},
            {'tool': 'dbt', 'action': 'run', 'args': {'model': 'clean_model'}},
            {'tool': 'airflow', 'action': 'trigger_dag', 'args': {'dag_id': 'finalize_data'}},
        ]
        manager.run_pipeline(elt_pipeline)
        # Step 5: Tune DB
        manager.tune_db('snowflake', {'indexes': True, 'partitions': ['date']})
        # Step 6: Version data
        manager.version_data()
        # Step 7: Dashboard/report
        manager.add_connector('powerbi', PowerBIConnector('workspace', 'token'))
        manager.dashboard('powerbi', {'title': 'Final Clean Data Report'})
        # Step 8: Integration test (assertions)
        assert manager.metrics['synthetic_data_generated'] == 1
        assert manager.metrics['uploads'] == 1
        assert manager.metrics['elt_runs'] == 1
        assert manager.metrics['db_tuned'] == 1
        assert manager.metrics['versioned'] == 1
        print('Integration test passed!')
        print('Governance logs:', manager.get_governance_logs())
        print('Metrics:', manager.get_metrics())

    print("\n--- Running comprehensive end-to-end pipeline example ---")
    example_end_to_end_pipeline()
