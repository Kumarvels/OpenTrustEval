"""
Dataset Management Integration for Data Engineering
Integrates dataset management features: dataset creation, processing, validation, visualization, export/import
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

import re
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

# Remove Cleanlab imports and logic except for benchmarking
# Remove: from data_engineering.cleanlab_integration import CleanlabDataQualityManager
# Remove: self.cleanlab_manager and all Cleanlab-based validation/filtering
# All validation, filtering, and reporting should use fallback or advanced trust scoring only

class DatasetManager:
    """
    Dataset Manager - Integrates dataset management features into data engineering workflow
    """
    
    def __init__(self, base_path: str = "./datasets"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.datasets = {}
        self.metadata = {}
        self.validation_results = {}
        self.processing_history = []
        self.logger = self._setup_logger()
        
        # Initialize Cleanlab integration
        # self.cleanlab_manager = CleanlabDataQualityManager() # Removed Cleanlab integration
        # if CLEANLAB_AVAILABLE:
        #     self.cleanlab_manager = CleanlabDataQualityManager()
        # else:
        #     self.cleanlab_manager = None
        #     self.logger.warning("Cleanlab not available. Advanced data quality features disabled.")
        
        # Load existing datasets from disk
        self._load_existing_datasets()
        
    def _setup_logger(self):
        """Setup logging for dataset operations"""
        logger = logging.getLogger('DatasetManager')
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
    
    def _load_dataset_metadata(self, dataset_id: str) -> Dict:
        """Load dataset metadata"""
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        dataset_info = self.datasets[dataset_id]
        metadata_path = Path(dataset_info['path']) / 'metadata.json'
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        else:
            # Return basic metadata if file doesn't exist
            return {
                'name': dataset_info.get('name', 'unknown'),
                'created_at': dataset_info.get('created_at', datetime.now().isoformat()),
                'rows': dataset_info.get('rows', 0),
                'columns': dataset_info.get('columns', 0)
            }

    def create_quality_filtered_dataset(self, dataset_id: str, min_trust_score: float = 0.7,
                                      features: Optional[List] = None) -> str:
        """
        Create a quality-filtered dataset using fallback trust scoring
        """
        try:
            df = self.load_dataset(dataset_id)
            
            # Use fallback quality manager
            from data_engineering.cleanlab_integration import FallbackDataQualityManager
            quality_manager = FallbackDataQualityManager()
            
            # Apply quality filtering
            filtered_df = quality_manager.create_quality_based_filter(
                df, min_trust_score=min_trust_score, features=features
            )
            
            # Create new dataset with filtered data
            original_metadata = self._load_dataset_metadata(dataset_id)
            new_name = f"{original_metadata.get('name', 'dataset')}_quality_filtered"
            
            new_dataset_id = self.create_dataset(
                name=new_name,
                data=filtered_df,
                schema=original_metadata.get('schema'),
                metadata={
                    **original_metadata,
                    'filtered_from': dataset_id,
                    'min_trust_score': min_trust_score,
                    'filtering_method': 'fallback_trust_scoring',
                    'original_rows': len(df),
                    'filtered_rows': len(filtered_df),
                    'retention_rate': len(filtered_df) / len(df) if len(df) > 0 else 0
                }
            )
            
            self.logger.info(f"Created quality-filtered dataset: {new_dataset_id}")
            return new_dataset_id
            
        except Exception as e:
            self.logger.error(f"Error creating quality-filtered dataset: {e}")
            raise
    
    def is_safe_query(self, query_str: str, allowed_columns) -> bool:
        """
        Check if the query string is safe: only allowed column names, numbers, and safe operators.
        """
        # Only allow column names, numbers, whitespace, and safe operators
        # Disallow parentheses, function calls, __import__, etc.
        # Allowed operators: ==, !=, <, >, <=, >=, and, or, not
        # Build regex for allowed columns
        col_pattern = r'|'.join([re.escape(col) for col in allowed_columns])
        # Full pattern: allowed columns, numbers, operators, whitespace
        safe_pattern = rf'^([\s\d\.\'"]*({col_pattern})[\s\d\.\'"]*(==|!=|<=|>=|<|>|and|or|not|&|\||\s)*[\s\d\.\'"]*)+$'
        # Disallow suspicious keywords
        forbidden = ['__import__', 'os.', 'sys.', 'eval', 'exec', 'open(', '(', ')', '[', ']', '{', '}', ';']
        lowered = query_str.lower()
        for word in forbidden:
            if word in lowered:
                return False
        # Check regex
        if re.match(safe_pattern, query_str):
            return True
        return False

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
                condition = params['condition']
                if not self.is_safe_query(condition, df.columns):
                    raise ValueError("Unsafe filter condition detected. Only simple column comparisons are allowed.")
                df = df.query(condition)
            elif operation == 'sort':
                df = df.sort_values(by=params['columns'], ascending=params.get('ascending', True))
            elif operation == 'groupby':
                group_cols = params['columns']
                agg_funcs = params.get('agg_funcs', {})
                df = df.groupby(group_cols).agg(agg_funcs).reset_index()
            elif operation == 'custom':
                # Custom transformation function
                func = params['function']
                df = func(df)
            else:
                raise ValueError(f"Unknown transformation operation: {operation}")
        
        # Create new dataset
        new_name = f"{original_dataset['name']}_processed"
        new_dataset_id = self.create_dataset(new_name, df)
        
        # Record processing history
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
        Create visualization for dataset
        
        Args:
            dataset_id: Dataset to visualize
            visualization_config: Configuration for visualization
            
        Returns:
            Path to saved visualization file
        """
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available for visualization")
            return None
        
        df = self.load_dataset(dataset_id)
        viz_type = visualization_config['type']
        
        if viz_type == 'scatter':
            fig = px.scatter(df, x=visualization_config['x'], y=visualization_config['y'])
        elif viz_type == 'histogram':
            fig = px.histogram(df, x=visualization_config['column'])
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
        else:
            output_path = Path(output_path)
        
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records', indent=2)
        elif format == 'parquet':
            if not PARQUET_AVAILABLE:
                raise ImportError("PyArrow required for Parquet export")
            df.to_parquet(output_path, index=False)
        elif format == 'excel':
            df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Exported dataset {dataset_id} to {output_path}")
        return str(output_path)
    
    def import_dataset(self, file_path: str, name: str, format: Optional[str] = None) -> str:
        """
        Import dataset from file
        
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
            raise ValueError(f"Unsupported import format: {format}")
        
        dataset_id = self.create_dataset(name, df)
        self.logger.info(f"Imported dataset '{name}' from {file_path}")
        return dataset_id
    
    def list_datasets(self) -> List[Dict]:
        """List all datasets with metadata"""
        datasets = []
        for dataset_id, info in self.datasets.items():
            # Load metadata
            metadata_file = info['metadata_file']
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            datasets.append({
                'id': dataset_id,
                'name': info['name'],
                'path': info['path'],
                'metadata': metadata
            })
        
        return datasets
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """
        Delete a dataset
        
        Args:
            dataset_id: Dataset to delete
            
        Returns:
            Success status
        """
        if dataset_id not in self.datasets:
            return False
        
        dataset_info = self.datasets[dataset_id]
        dataset_path = Path(dataset_info['path'])
        
        # Remove directory and all files
        import shutil
        try:
            shutil.rmtree(dataset_path)
            del self.datasets[dataset_id]
            self.logger.info(f"Deleted dataset {dataset_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting dataset {dataset_id}: {e}")
            return False
    
    def _generate_dataset_id(self, name: str) -> str:
        """Generate unique dataset ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
        return f"{name_hash}_{timestamp}"
    
    def _infer_schema(self, df: pd.DataFrame) -> Dict:
        """Infer schema from DataFrame"""
        schema = {}
        for col in df.columns:
            schema[col] = {
                'type': str(df[col].dtype),
                'nullable': bool(df[col].isnull().any()),  # Convert to native Python bool
                'unique': bool(df[col].is_unique)  # Convert to native Python bool
            }
        return schema
    
    def _type_matches(self, expected: str, actual: str) -> bool:
        """Check if actual type matches expected type"""
        type_mapping = {
            'int64': ['int64', 'int32', 'int'],
            'float64': ['float64', 'float32', 'float'],
            'string': ['object', 'string'],
            'bool': ['bool', 'boolean']
        }
        
        for expected_type, actual_types in type_mapping.items():
            if expected in actual_types:
                return actual in actual_types
        return expected == actual
    
    def _load_existing_datasets(self):
        """Load existing datasets from disk"""
        if not self.base_path.exists():
            return
        
        for dataset_dir in self.base_path.iterdir():
            if dataset_dir.is_dir():
                metadata_file = dataset_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        # Find the dataset name from metadata or use directory name
                        dataset_name = metadata.get('name', dataset_dir.name)
                        
                        self.datasets[dataset_dir.name] = {
                            'name': dataset_name,
                            'path': str(dataset_dir),
                            'data_file': str(dataset_dir / "data.csv"),
                            'schema_file': str(dataset_dir / "schema.json"),
                            'metadata_file': str(metadata_file)
                        }
                    except Exception as e:
                        self.logger.warning(f"Error loading dataset {dataset_dir.name}: {e}")

class DatasetConnector:
    """Connector class for integration with DataLifecycleManager"""
    
    def __init__(self, base_path: str = "./datasets"):
        self.dataset_manager = DatasetManager(base_path)
    
    def create_dataset(self, name: str, data: Union[pd.DataFrame, Dict, List], 
                      schema: Optional[Dict] = None, metadata: Optional[Dict] = None) -> str:
        return self.dataset_manager.create_dataset(name, data, schema, metadata)
    
    def load_dataset(self, dataset_id: str) -> pd.DataFrame:
        return self.dataset_manager.load_dataset(dataset_id)
    
    def validate_dataset(self, dataset_id: str, validation_rules: Optional[Dict] = None) -> Dict:
        return self.dataset_manager.validate_dataset(dataset_id, validation_rules)
    
    def create_quality_filtered_dataset(self, dataset_id: str, min_trust_score: float = 0.7,
                                      features: Optional[List] = None) -> str:
        return self.dataset_manager.create_quality_filtered_dataset(dataset_id, min_trust_score, features)
    
    def process_dataset(self, dataset_id: str, transformations: List[Dict]) -> str:
        return self.dataset_manager.process_dataset(dataset_id, transformations)
    
    def visualize_dataset(self, dataset_id: str, visualization_config: Dict) -> Optional[str]:
        return self.dataset_manager.visualize_dataset(dataset_id, visualization_config)
    
    def export_dataset(self, dataset_id: str, format: str, output_path: Optional[str] = None) -> str:
        return self.dataset_manager.export_dataset(dataset_id, format, output_path)
    
    def import_dataset(self, file_path: str, name: str, format: Optional[str] = None) -> str:
        return self.dataset_manager.import_dataset(file_path, name, format)
    
    def list_datasets(self) -> List[Dict]:
        return self.dataset_manager.list_datasets()
    
    def delete_dataset(self, dataset_id: str) -> bool:
        return self.dataset_manager.delete_dataset(dataset_id)

def example_dataset_integration():
    """Example usage of dataset management features with Cleanlab integration"""
    
    # Initialize dataset manager
    dataset_manager = DatasetManager()
    
    # Create sample data with some quality issues
    sample_data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry', 'Ivy', 'Jack'],
        'age': [25, 30, 35, 28, 32, 45, 29, 38, 27, 33],
        'salary': [50000, 60000, 70000, 55000, 65000, 80000, 52000, 72000, 48000, 68000],
        'department': ['IT', 'HR', 'IT', 'Finance', 'IT', 'HR', 'Finance', 'IT', 'HR', 'IT']
    }
    
    # Create dataset
    dataset_id = dataset_manager.create_dataset('employees', sample_data)
    print(f"Created dataset: {dataset_id}")
    
    # Standard validation
    validation_results = dataset_manager.validate_dataset(dataset_id)
    print(f"Standard Validation: {'PASSED' if validation_results['passed'] else 'FAILED'}")
    if validation_results['warnings']:
        print("Warnings:", validation_results['warnings'])
    
    # Cleanlab validation (if available)
    # if dataset_manager.cleanlab_manager: # Removed Cleanlab integration
    #     print("\n=== Cleanlab Data Quality Assessment ===") # Removed Cleanlab integration
    #     cleanlab_results = dataset_manager.validate_dataset_with_cleanlab(dataset_id) # Removed Cleanlab integration
        
    #     if 'error' not in cleanlab_results: # Removed Cleanlab integration
    #         trust_score = cleanlab_results.get('trust_score', 0) # Removed Cleanlab integration
    #         print(f"Data Trust Score: {trust_score:.3f}") # Removed Cleanlab integration
            
    #         # Create quality-filtered dataset # Removed Cleanlab integration
    #         if trust_score < 0.8:  # If trust score is low # Removed Cleanlab integration
    #             print("Creating quality-filtered dataset...") # Removed Cleanlab integration
    #             filtered_dataset_id = dataset_manager.create_quality_filtered_dataset( # Removed Cleanlab integration
    #                 dataset_id, min_trust_score=0.7 # Removed Cleanlab integration
    #             ) # Removed Cleanlab integration
    #             print(f"Quality-filtered dataset: {filtered_dataset_id}") # Removed Cleanlab integration
    #     else: # Removed Cleanlab integration
    #         print(f"Cleanlab error: {cleanlab_results['error']}") # Removed Cleanlab integration
    # else: # Removed Cleanlab integration
    #     print("Cleanlab not available for advanced data quality assessment") # Removed Cleanlab integration
    
    # Process dataset
    transformations = [
        {'operation': 'filter', 'params': {'condition': 'age > 30'}},
        {'operation': 'sort', 'params': {'columns': ['salary'], 'ascending': False}}
    ]
    processed_dataset_id = dataset_manager.process_dataset(dataset_id, transformations)
    print(f"Processed dataset: {processed_dataset_id}")
    
    # Visualize dataset
    viz_config = {
        'type': 'scatter',
        'x': 'age',
        'y': 'salary',
        'save': True
    }
    viz_path = dataset_manager.visualize_dataset(dataset_id, viz_config)
    print(f"Visualization saved: {viz_path}")
    
    # Export dataset
    export_path = dataset_manager.export_dataset(dataset_id, 'json')
    print(f"Exported to: {export_path}")
    
    # List all datasets
    datasets = dataset_manager.list_datasets()
    print(f"Total datasets: {len(datasets)}")
    
    return dataset_manager

if __name__ == "__main__":
    dataset_manager = example_dataset_integration() 