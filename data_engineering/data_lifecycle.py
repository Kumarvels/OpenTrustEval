import os
import sys
import pandas as pd
import numpy as np
import yaml
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging

# Import connectors
from data_engineering.connectors.spark_connector import SparkConnector
from data_engineering.connectors.dbt_connector import DBTConnector
from data_engineering.connectors.airflow_connector import AirflowConnector
from data_engineering.connectors.snowflake_connector import SnowflakeConnector
from data_engineering.connectors.powerbi_connector import PowerBIConnector
from data_engineering.connectors.bigquery_connector import BigQueryConnector
from data_engineering.connectors.redshift_connector import RedshiftConnector
from data_engineering.connectors.databricks_connector import DatabricksConnector
from data_engineering.connectors.kafka_connector import KafkaConnector
from data_engineering.connectors.mongodb_connector import MongoDBConnector
from data_engineering.connectors.postgresql_connector import PostgreSQLConnector
from data_engineering.connectors.hive_connector import HiveConnector
from data_engineering.connectors.flink_connector import FlinkConnector
from data_engineering.connectors.hudi_connector import HudiConnector
from data_engineering.connectors.looker_connector import LookerConnector
from data_engineering.connectors.fivetran_connector import FivetranConnector
from data_engineering.connectors.athena_connector import AthenaConnector
from data_engineering.connectors.synapse_connector import SynapseConnector

# Import dataset management
try:
    from data_engineering.dataset_integration import DatasetConnector
    DATASET_AVAILABLE = True
except ImportError:
    DATASET_AVAILABLE = False

class DataLifecycleManager:
    """
    Data Engineering Lifecycle Management
    - Data ingestion, processing, validation, and monitoring
    - Connector management for various data sources
    - Dataset management with validation and processing
    - Governance and audit logging
    - Metrics collection
    """
    
    def __init__(self):
        self.connectors = {}
        self.databases = {}
        self.metrics = {}
        self.governance_logs = []
        self.dataset_manager = None
        
        # Initialize dataset management if available
        if DATASET_AVAILABLE:
            self.dataset_manager = DatasetConnector()
            self.add_connector('dataset', self.dataset_manager)
        
        # Initialize logging
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup logging for data lifecycle operations"""
        logger = logging.getLogger('DataLifecycleManager')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def add_connector(self, name: str, connector):
        """Add a data connector (Spark, dbt, Airflow, etc.)"""
        self.connectors[name] = connector
        self.log_governance(f'Connector added: {name}')
    
    def add_db(self, name: str, db_connector):
        """Add a database connector (Snowflake, BigQuery, etc.)"""
        self.databases[name] = db_connector
        self.log_governance(f'Database added: {name}')
    
    def run_pipeline(self, pipeline_steps: List[Dict]):
        """Run a data pipeline with multiple steps"""
        for step in pipeline_steps:
            tool = step['tool']
            action = step['action']
            args = step.get('args', {})
            
            if tool in self.connectors:
                connector = self.connectors[tool]
                if hasattr(connector, action):
                    getattr(connector, action)(**args)
                else:
                    self.logger.warning(f"Action {action} not found in connector {tool}")
            else:
                self.logger.error(f"Connector {tool} not found")
        
        self.metrics['pipelines_run'] = self.metrics.get('pipelines_run', 0) + 1
        self.log_governance('Pipeline executed')
    
    def generate_synthetic_data(self, config: Dict):
        """Generate synthetic data for testing"""
        rows = config.get('rows', 100)
        schema = config.get('schema', {})
        
        # Generate synthetic data based on schema
        data = {}
        for col, col_config in schema.items():
            col_type = col_config.get('type', 'string')
            if 'int' in col_type:
                data[col] = np.random.randint(1, 1000, rows)
            elif 'float' in col_type:
                data[col] = np.random.rand(rows) * 1000
            elif 'string' in col_type or 'object' in col_type:
                data[col] = [f"value_{i}" for i in range(rows)]
            elif 'bool' in col_type:
                data[col] = np.random.choice([True, False], rows)
        
        df = pd.DataFrame(data)
        self.metrics['synthetic_data_generated'] = self.metrics.get('synthetic_data_generated', 0) + 1
        self.log_governance('Synthetic data generated')
        return df
    
    def upload_data(self, source_type: str, source_path: str):
        """Upload data to a destination"""
        self.metrics['data_uploads'] = self.metrics.get('data_uploads', 0) + 1
        self.log_governance(f'Data uploaded from {source_type}: {source_path}')
    
    def sql_transform(self, db_name: str, query: str):
        """Execute SQL transformation on a database"""
        if db_name in self.databases:
            db = self.databases[db_name]
            # Simulate SQL execution
            self.metrics['sql_transforms'] = self.metrics.get('sql_transforms', 0) + 1
            self.log_governance(f'SQL transform executed on {db_name}')
            return pd.DataFrame()  # Return empty DataFrame for demo
        else:
            self.logger.error(f"Database {db_name} not found")
    
    def dashboard(self, tool: str, config: Dict):
        """Create dashboard/report"""
        self.metrics['dashboards_created'] = self.metrics.get('dashboards_created', 0) + 1
        self.log_governance(f'Dashboard created with {tool}')
    
    def log_governance(self, action: str):
        """Log governance/audit actions"""
        self.governance_logs.append({
            'timestamp': datetime.now().isoformat(),
            'action': action
        })
    
    def get_metrics(self):
        """Get collected metrics"""
        return self.metrics
    
    def get_governance_logs(self):
        """Get governance/audit logs"""
        return self.governance_logs
    
    # Dataset Management Methods
    def create_dataset(self, name: str, data: Union[pd.DataFrame, Dict, List], 
                      schema: Optional[Dict] = None, metadata: Optional[Dict] = None) -> str:
        """Create a new dataset"""
        if not self.dataset_manager:
            raise RuntimeError("Dataset management not available")
        
        dataset_id = self.dataset_manager.create_dataset(name, data, schema, metadata)
        self.metrics['datasets_created'] = self.metrics.get('datasets_created', 0) + 1
        self.log_governance(f'Dataset created: {name}')
        return dataset_id
    
    def load_dataset(self, dataset_id: str) -> pd.DataFrame:
        """Load a dataset by ID"""
        if not self.dataset_manager:
            raise RuntimeError("Dataset management not available")
        
        df = self.dataset_manager.load_dataset(dataset_id)
        self.metrics['datasets_loaded'] = self.metrics.get('datasets_loaded', 0) + 1
        self.log_governance(f'Dataset loaded: {dataset_id}')
        return df
    
    def validate_dataset(self, dataset_id: str, validation_rules=None):
        """Validate a dataset"""
        if not self.dataset_manager:
            raise RuntimeError("Dataset management not available")
        
        results = self.dataset_manager.validate_dataset(dataset_id, validation_rules)
        self.metrics['datasets_validated'] = self.metrics.get('datasets_validated', 0) + 1
        self.log_governance(f'Dataset validated: {dataset_id} ({results["passed"]})')
        return results
    
    def process_dataset(self, dataset_id: str, transformations):
        """Process a dataset with transformations"""
        if not self.dataset_manager:
            raise RuntimeError("Dataset management not available")
        
        new_dataset_id = self.dataset_manager.process_dataset(dataset_id, transformations)
        self.metrics['datasets_processed'] = self.metrics.get('datasets_processed', 0) + 1
        self.log_governance(f'Dataset processed: {dataset_id} -> {new_dataset_id}')
        return new_dataset_id
    
    def visualize_dataset(self, dataset_id: str, visualization_config):
        """Create visualization for dataset"""
        if not self.dataset_manager:
            raise RuntimeError("Dataset management not available")
        
        viz_path = self.dataset_manager.visualize_dataset(dataset_id, visualization_config)
        self.metrics['visualizations_created'] = self.metrics.get('visualizations_created', 0) + 1
        self.log_governance(f'Visualization created: {dataset_id}')
        return viz_path
    
    def export_dataset(self, dataset_id: str, format_type, output_path=None):
        """Export dataset to various formats"""
        if not self.dataset_manager:
            raise RuntimeError("Dataset management not available")
        
        export_path = self.dataset_manager.export_dataset(dataset_id, format_type, output_path)
        self.metrics['datasets_exported'] = self.metrics.get('datasets_exported', 0) + 1
        self.log_governance(f'Dataset exported: {dataset_id} -> {export_path}')
        return export_path
    
    def import_dataset(self, file_path: str, name: str, format_type=None):
        """Import dataset from file"""
        if not self.dataset_manager:
            raise RuntimeError("Dataset management not available")
        
        dataset_id = self.dataset_manager.import_dataset(file_path, name, format_type)
        self.metrics['datasets_imported'] = self.metrics.get('datasets_imported', 0) + 1
        self.log_governance(f'Dataset imported: {file_path} -> {dataset_id}')
        return dataset_id
    
    def list_datasets(self):
        """List all datasets"""
        if not self.dataset_manager:
            raise RuntimeError("Dataset management not available")
        
        datasets = self.dataset_manager.list_datasets()
        self.log_governance(f'Datasets listed: {len(datasets)} found')
        return datasets
    
    def delete_dataset(self, dataset_id: str):
        """Delete a dataset"""
        if not self.dataset_manager:
            raise RuntimeError("Dataset management not available")
        
        success = self.dataset_manager.delete_dataset(dataset_id)
        if success:
            self.metrics['datasets_deleted'] = self.metrics.get('datasets_deleted', 0) + 1
            self.log_governance(f'Dataset deleted: {dataset_id}')
        return success
