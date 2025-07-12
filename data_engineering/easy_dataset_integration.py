"""
Easy Dataset Integration for Data Engineering
Integrates Easy Dataset features: dataset management, processing, validation, visualization, export/import
Reference: https://github.com/ConardLi/easy-dataset
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import yaml
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging
from datetime import datetime
import hashlib

# Optional imports for advanced features
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import great_expectations as ge
    GE_AVAILABLE = True
except ImportError:
    GE_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

class EasyDatasetManager:
    """
    Easy Dataset Manager - Integrates Easy Dataset features into data engineering workflow
    """
    
    def __init__(self, base_path: str = "./datasets"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.datasets = {}
        self.metadata = {}
        self.validation_results = {}
        self.processing_history = []
        self.logger = self._setup_logger()
        # Load existing datasets from disk
        self._load_existing_datasets()
        
    def _setup_logger(self):
        """Setup logging for dataset operations"""
        logger = logging.getLogger('EasyDatasetManager')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def create_dataset(self, name: str, data: Union[pd.DataFrame, Dict, List], 
                      schema: Optional[Dict] = None, metadata: Optional[Dict] = None) -> str:
        """
        Create a new dataset with validation and metadata
        
        Args:
            name: Dataset name
            data: Data as DataFrame, dict, or list
            schema: Optional schema definition
            metadata: Optional metadata
            
        Returns:
            Dataset ID
        """
        dataset_id = self._generate_dataset_id(name)
        dataset_path = self.base_path / f"{dataset_id}"
        dataset_path.mkdir(exist_ok=True)
        
        # Convert data to DataFrame if needed
        if isinstance(data, (dict, list)):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Save data
        data_file = dataset_path / "data.csv"
        df.to_csv(data_file, index=False)
        
        # Generate schema if not provided
        if schema is None:
            schema = self._infer_schema(df)
        
        # Save schema
        schema_file = dataset_path / "schema.json"
        with open(schema_file, 'w') as f:
            json.dump(schema, f, indent=2)
        
        # Save metadata
        metadata = metadata or {}
        metadata.update({
            'created_at': datetime.now().isoformat(),
            'rows': len(df),
            'columns': len(df.columns),
            'size_mb': data_file.stat().st_size / (1024 * 1024)
        })
        
        metadata_file = dataset_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Store in memory
        self.datasets[dataset_id] = {
            'name': name,
            'path': str(dataset_path),
            'data_file': str(data_file),
            'schema_file': str(schema_file),
            'metadata_file': str(metadata_file)
        }
        
        self.logger.info(f"Created dataset '{name}' with ID: {dataset_id}")
        return dataset_id
    
    def load_dataset(self, dataset_id: str) -> pd.DataFrame:
        """Load a dataset by ID"""
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        dataset_info = self.datasets[dataset_id]
        data_file = dataset_info['data_file']
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        df = pd.read_csv(data_file)
        self.logger.info(f"Loaded dataset {dataset_id} with {len(df)} rows")
        return df
    
    def validate_dataset(self, dataset_id: str, validation_rules: Optional[Dict] = None) -> Dict:
        """
        Validate dataset against schema and custom rules
        
        Args:
            dataset_id: Dataset to validate
            validation_rules: Optional custom validation rules
            
        Returns:
            Validation results
        """
        df = self.load_dataset(dataset_id)
        dataset_info = self.datasets[dataset_id]
        
        # Load schema
        with open(dataset_info['schema_file'], 'r') as f:
            schema = json.load(f)
        
        validation_results = {
            'dataset_id': dataset_id,
            'timestamp': datetime.now().isoformat(),
            'passed': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Schema validation
        for col, col_schema in schema.items():
            if col not in df.columns:
                validation_results['errors'].append(f"Missing column: {col}")
                validation_results['passed'] = False
                continue
            
            # Type validation
            expected_type = col_schema.get('type')
            if expected_type:
                actual_type = str(df[col].dtype)
                if not self._type_matches(expected_type, actual_type):
                    validation_results['warnings'].append(
                        f"Column {col}: expected {expected_type}, got {actual_type}"
                    )
            
            # Null check
            null_count = df[col].isnull().sum()
            if null_count > 0:
                validation_results['warnings'].append(
                    f"Column {col}: {null_count} null values found"
                )
            
            # Unique check
            if col_schema.get('unique', False):
                if not df[col].is_unique:
                    validation_results['errors'].append(f"Column {col}: not unique")
                    validation_results['passed'] = False
        
        # Custom validation rules
        if validation_rules:
            for rule_name, rule_func in validation_rules.items():
                try:
                    result = rule_func(df)
                    if not result:
                        validation_results['errors'].append(f"Custom rule failed: {rule_name}")
                        validation_results['passed'] = False
                except Exception as e:
                    validation_results['errors'].append(f"Custom rule error {rule_name}: {str(e)}")
                    validation_results['passed'] = False
        
        # Basic stats
        validation_results['stats'] = {
            'rows': int(len(df)),
            'columns': int(len(df.columns)),
            'null_counts': {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()},
            'duplicate_rows': int(df.duplicated().sum())
        }
        
        self.validation_results[dataset_id] = validation_results
        self.logger.info(f"Validation completed for {dataset_id}: {'PASSED' if validation_results['passed'] else 'FAILED'}")
        
        return validation_results
    
    def process_dataset(self, dataset_id: str, transformations: List[Dict]) -> str:
        """
        Apply transformations to a dataset
        
        Args:
            dataset_id: Dataset to process
            transformations: List of transformation operations
            
        Returns:
            New dataset ID
        """
        df = self.load_dataset(dataset_id)
        original_dataset = self.datasets[dataset_id]
        
        # Apply transformations
        for transform in transformations:
            operation = transform['operation']
            params = transform.get('params', {})
            
            if operation == 'drop_columns':
                df = df.drop(columns=params['columns'])
            elif operation == 'rename_columns':
                df = df.rename(columns=params['mapping'])
            elif operation == 'filter':
                df = df.query(params['condition'])
            elif operation == 'sort':
                df = df.sort_values(by=params['columns'], ascending=params.get('ascending', True))
            elif operation == 'groupby':
                grouped = df.groupby(params['columns']).agg(params['aggregations'])
                df = grouped.reset_index()
            elif operation == 'custom':
                # Custom transformation function
                custom_func = params['function']
                df = custom_func(df)
            else:
                raise ValueError(f"Unknown transformation: {operation}")
        
        # Create new dataset
        new_name = f"{original_dataset['name']}_processed"
        new_dataset_id = self.create_dataset(new_name, df)
        
        # Log processing history
        self.processing_history.append({
            'original_dataset': dataset_id,
            'new_dataset': new_dataset_id,
            'transformations': transformations,
            'timestamp': datetime.now().isoformat()
        })
        
        self.logger.info(f"Processed dataset {dataset_id} -> {new_dataset_id}")
        return new_dataset_id
    
    def visualize_dataset(self, dataset_id: str, visualization_config: Dict) -> Optional[str]:
        """
        Create visualizations for the dataset
        
        Args:
            dataset_id: Dataset to visualize
            visualization_config: Configuration for visualization
            
        Returns:
            Path to saved visualization (if saved)
        """
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available for visualization")
            return None
        
        df = self.load_dataset(dataset_id)
        viz_type = visualization_config['type']
        
        if viz_type == 'histogram':
            fig = px.histogram(df, x=visualization_config['column'])
        elif viz_type == 'scatter':
            fig = px.scatter(df, x=visualization_config['x'], y=visualization_config['y'])
        elif viz_type == 'bar':
            fig = px.bar(df, x=visualization_config['x'], y=visualization_config['y'])
        elif viz_type == 'line':
            fig = px.line(df, x=visualization_config['x'], y=visualization_config['y'])
        elif viz_type == 'correlation':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto")
        else:
            raise ValueError(f"Unknown visualization type: {viz_type}")
        
        # Save if requested
        if visualization_config.get('save', False):
            dataset_info = self.datasets[dataset_id]
            viz_path = Path(dataset_info['path']) / f"visualization_{viz_type}.html"
            fig.write_html(str(viz_path))
            self.logger.info(f"Visualization saved to {viz_path}")
            return str(viz_path)
        
        # Show if in interactive environment
        try:
            fig.show()
        except:
            pass
        
        return None
    
    def export_dataset(self, dataset_id: str, format: str, output_path: Optional[str] = None) -> str:
        """
        Export dataset to various formats
        
        Args:
            dataset_id: Dataset to export
            format: Export format (csv, json, parquet, excel)
            output_path: Optional output path
            
        Returns:
            Path to exported file
        """
        df = self.load_dataset(dataset_id)
        dataset_info = self.datasets[dataset_id]
        
        if output_path is None:
            output_path = Path(dataset_info['path']) / f"export.{format}"
        
        if format.lower() == 'csv':
            df.to_csv(output_path, index=False)
        elif format.lower() == 'json':
            df.to_json(output_path, orient='records', indent=2)
        elif format.lower() == 'parquet':
            if not PARQUET_AVAILABLE:
                raise ImportError("PyArrow required for Parquet export")
            df.to_parquet(output_path, index=False)
        elif format.lower() == 'excel':
            df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Exported dataset {dataset_id} to {output_path}")
        return str(output_path)
    
    def import_dataset(self, file_path: str, name: str, format: Optional[str] = None) -> str:
        """
        Import dataset from various formats
        
        Args:
            file_path: Path to file
            name: Dataset name
            format: File format (auto-detected if None)
            
        Returns:
            Dataset ID
        """
        file_path = Path(file_path)
        
        if format is None:
            format = file_path.suffix.lower().lstrip('.')
        
        if format == 'csv':
            df = pd.read_csv(file_path)
        elif format == 'json':
            df = pd.read_json(file_path)
        elif format == 'parquet':
            if not PARQUET_AVAILABLE:
                raise ImportError("PyArrow required for Parquet import")
            df = pd.read_parquet(file_path)
        elif format == 'excel':
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        dataset_id = self.create_dataset(name, df)
        self.logger.info(f"Imported dataset '{name}' from {file_path}")
        return dataset_id
    
    def list_datasets(self) -> List[Dict]:
        """List all available datasets"""
        datasets = []
        for dataset_id, info in self.datasets.items():
            try:
                with open(info['metadata_file'], 'r') as f:
                    metadata = json.load(f)
                datasets.append({
                    'id': dataset_id,
                    'name': info['name'],
                    'path': info['path'],
                    'metadata': metadata
                })
            except:
                datasets.append({
                    'id': dataset_id,
                    'name': info['name'],
                    'path': info['path'],
                    'metadata': {}
                })
        return datasets
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset"""
        if dataset_id not in self.datasets:
            return False
        
        dataset_info = self.datasets[dataset_id]
        dataset_path = Path(dataset_info['path'])
        
        # Remove files
        for file_path in dataset_path.glob('*'):
            file_path.unlink()
        
        # Remove directory
        dataset_path.rmdir()
        
        # Remove from memory
        del self.datasets[dataset_id]
        
        self.logger.info(f"Deleted dataset {dataset_id}")
        return True
    
    def _generate_dataset_id(self, name: str) -> str:
        """Generate unique dataset ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
        return f"{name_hash}_{timestamp}"
    
    def _infer_schema(self, df: pd.DataFrame) -> Dict:
        """Infer schema from DataFrame"""
        schema = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            schema[col] = {
                'type': dtype,
                'nullable': bool(df[col].isnull().any()),
                'unique': bool(df[col].is_unique)
            }
        return schema
    
    def _type_matches(self, expected: str, actual: str) -> bool:
        """Check if actual type matches expected type"""
        type_mapping = {
            'int64': 'int',
            'int32': 'int',
            'float64': 'float',
            'float32': 'float',
            'object': 'string',
            'string': 'string',
            'bool': 'boolean'
        }
        return type_mapping.get(actual, actual) == expected

    def _load_existing_datasets(self):
        """Load existing datasets from disk on initialization"""
        if not self.base_path.exists():
            return
        
        for dataset_dir in self.base_path.iterdir():
            if dataset_dir.is_dir():
                metadata_file = dataset_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        # Extract dataset name from metadata or directory name
                        dataset_name = metadata.get('name', dataset_dir.name)
                        
                        # Find the data file
                        data_file = None
                        for file in dataset_dir.glob("*.csv"):
                            data_file = file
                            break
                        
                        if data_file and data_file.exists():
                            self.datasets[dataset_dir.name] = {
                                'name': dataset_name,
                                'path': str(dataset_dir),
                                'data_file': str(data_file),
                                'schema_file': str(dataset_dir / "schema.json"),
                                'metadata_file': str(metadata_file)
                            }
                            self.logger.info(f"Loaded existing dataset: {dataset_name} ({dataset_dir.name})")
                    except Exception as e:
                        self.logger.warning(f"Failed to load dataset from {dataset_dir}: {e}")


# Integration with existing DataLifecycleManager
class EasyDatasetConnector:
    """Connector to integrate Easy Dataset with DataLifecycleManager"""
    
    def __init__(self, base_path: str = "./datasets"):
        self.easy_dataset = EasyDatasetManager(base_path)
    
    def create_dataset(self, name: str, data: Union[pd.DataFrame, Dict, List], 
                      schema: Optional[Dict] = None, metadata: Optional[Dict] = None) -> str:
        return self.easy_dataset.create_dataset(name, data, schema, metadata)
    
    def load_dataset(self, dataset_id: str) -> pd.DataFrame:
        return self.easy_dataset.load_dataset(dataset_id)
    
    def validate_dataset(self, dataset_id: str, validation_rules: Optional[Dict] = None) -> Dict:
        return self.easy_dataset.validate_dataset(dataset_id, validation_rules)
    
    def process_dataset(self, dataset_id: str, transformations: List[Dict]) -> str:
        return self.easy_dataset.process_dataset(dataset_id, transformations)
    
    def visualize_dataset(self, dataset_id: str, visualization_config: Dict) -> Optional[str]:
        return self.easy_dataset.visualize_dataset(dataset_id, visualization_config)
    
    def export_dataset(self, dataset_id: str, format: str, output_path: Optional[str] = None) -> str:
        return self.easy_dataset.export_dataset(dataset_id, format, output_path)
    
    def import_dataset(self, file_path: str, name: str, format: Optional[str] = None) -> str:
        return self.easy_dataset.import_dataset(file_path, name, format)
    
    def list_datasets(self) -> List[Dict]:
        return self.easy_dataset.list_datasets()
    
    def delete_dataset(self, dataset_id: str) -> bool:
        return self.easy_dataset.delete_dataset(dataset_id)


# Example usage and integration
def example_easy_dataset_integration():
    """Example of using Easy Dataset features"""
    
    # Create Easy Dataset manager
    easy_dataset = EasyDatasetManager()
    
    # Create sample dataset
    sample_data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'salary': [50000, 60000, 70000, 55000, 65000]
    }
    
    dataset_id = easy_dataset.create_dataset('employees', sample_data)
    print(f"Created dataset: {dataset_id}")
    
    # Validate dataset
    validation_results = easy_dataset.validate_dataset(dataset_id)
    print(f"Validation: {'PASSED' if validation_results['passed'] else 'FAILED'}")
    
    # Process dataset
    transformations = [
        {'operation': 'filter', 'params': {'condition': 'age > 30'}},
        {'operation': 'sort', 'params': {'columns': ['salary'], 'ascending': False}}
    ]
    
    processed_dataset_id = easy_dataset.process_dataset(dataset_id, transformations)
    print(f"Processed dataset: {processed_dataset_id}")
    
    # Visualize dataset
    viz_config = {
        'type': 'scatter',
        'x': 'age',
        'y': 'salary',
        'save': True
    }
    
    viz_path = easy_dataset.visualize_dataset(dataset_id, viz_config)
    if viz_path:
        print(f"Visualization saved: {viz_path}")
    
    # Export dataset
    export_path = easy_dataset.export_dataset(dataset_id, 'json')
    print(f"Exported to: {export_path}")
    
    # List all datasets
    datasets = easy_dataset.list_datasets()
    print(f"Available datasets: {len(datasets)}")
    
    return easy_dataset


if __name__ == "__main__":
    # Run example
    easy_dataset = example_easy_dataset_integration() 