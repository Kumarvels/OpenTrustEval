#!/usr/bin/env python3
"""
Easy Dataset CLI - Command-line interface for Easy Dataset management
Provides access to all Easy Dataset features: create, validate, process, visualize, export, import, etc.
"""

import os
import sys
import argparse
import json
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from data_engineering.easy_dataset_integration import EasyDatasetManager

def main():
    parser = argparse.ArgumentParser(
        description="Easy Dataset CLI - Manage datasets with validation, processing, visualization, and export/import",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a dataset from CSV
  python easy_dataset_cli.py create --name "my_dataset" --input data.csv --format csv
  
  # Validate a dataset
  python easy_dataset_cli.py validate --id dataset_123
  
  # Process a dataset with transformations
  python easy_dataset_cli.py process --id dataset_123 --transformations '[{"operation": "filter", "params": {"condition": "age > 30"}}]'
  
  # Visualize a dataset
  python easy_dataset_cli.py visualize --id dataset_123 --type scatter --x age --y salary --save
  
  # Export a dataset
  python easy_dataset_cli.py export --id dataset_123 --format json --output data.json
  
  # Import a dataset
  python easy_dataset_cli.py import --name "new_dataset" --input data.json --format json
  
  # List all datasets
  python easy_dataset_cli.py list
  
  # Delete a dataset
  python easy_dataset_cli.py delete --id dataset_123
        """
    )
    
    parser.add_argument('--base-path', default='./datasets', help='Base path for datasets (relative to data_engineering directory)')
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Create dataset
    create_parser = subparsers.add_parser('create', help='Create a new dataset')
    create_parser.add_argument('--name', required=True, help='Dataset name')
    create_parser.add_argument('--input', required=True, help='Input file path')
    create_parser.add_argument('--format', default='csv', choices=['csv', 'json', 'parquet', 'excel'], help='Input format')
    create_parser.add_argument('--schema', help='Schema file path (JSON)')
    create_parser.add_argument('--metadata', help='Metadata file path (JSON)')
    
    # Load dataset
    load_parser = subparsers.add_parser('load', help='Load and display dataset')
    load_parser.add_argument('--id', required=True, help='Dataset ID')
    load_parser.add_argument('--head', type=int, default=5, help='Number of rows to display')
    load_parser.add_argument('--info', action='store_true', help='Show dataset info')
    
    # Validate dataset
    validate_parser = subparsers.add_parser('validate', help='Validate a dataset')
    validate_parser.add_argument('--id', required=True, help='Dataset ID')
    validate_parser.add_argument('--rules', help='Validation rules file path (JSON)')
    validate_parser.add_argument('--output', help='Output validation results to file')
    
    # Process dataset
    process_parser = subparsers.add_parser('process', help='Process a dataset with transformations')
    process_parser.add_argument('--id', required=True, help='Dataset ID')
    process_parser.add_argument('--transformations', required=True, help='JSON string of transformations')
    process_parser.add_argument('--output-id', help='Output dataset ID (auto-generated if not provided)')
    
    # Visualize dataset
    visualize_parser = subparsers.add_parser('visualize', help='Create visualizations for dataset')
    visualize_parser.add_argument('--id', required=True, help='Dataset ID')
    visualize_parser.add_argument('--type', required=True, choices=['histogram', 'scatter', 'bar', 'line', 'correlation'], help='Visualization type')
    visualize_parser.add_argument('--x', help='X-axis column')
    visualize_parser.add_argument('--y', help='Y-axis column')
    visualize_parser.add_argument('--column', help='Column for histogram')
    visualize_parser.add_argument('--save', action='store_true', help='Save visualization to file')
    visualize_parser.add_argument('--output', help='Output file path')
    
    # Export dataset
    export_parser = subparsers.add_parser('export', help='Export dataset to various formats')
    export_parser.add_argument('--id', required=True, help='Dataset ID')
    export_parser.add_argument('--format', required=True, choices=['csv', 'json', 'parquet', 'excel'], help='Export format')
    export_parser.add_argument('--output', help='Output file path')
    
    # Import dataset
    import_parser = subparsers.add_parser('import', help='Import dataset from various formats')
    import_parser.add_argument('--name', required=True, help='Dataset name')
    import_parser.add_argument('--input', required=True, help='Input file path')
    import_parser.add_argument('--format', choices=['csv', 'json', 'parquet', 'excel'], help='Input format (auto-detected if not provided)')
    
    # List datasets
    list_parser = subparsers.add_parser('list', help='List all available datasets')
    list_parser.add_argument('--format', choices=['table', 'json'], default='table', help='Output format')
    
    # Delete dataset
    delete_parser = subparsers.add_parser('delete', help='Delete a dataset')
    delete_parser.add_argument('--id', required=True, help='Dataset ID')
    delete_parser.add_argument('--force', action='store_true', help='Force deletion without confirmation')
    
    args = parser.parse_args()
    
    # Initialize Easy Dataset manager
    easy_dataset = EasyDatasetManager(args.base_path)
    
    try:
        if args.command == 'create':
            # Load schema and metadata if provided
            schema = None
            if args.schema:
                with open(args.schema, 'r') as f:
                    schema = json.load(f)
            
            metadata = None
            if args.metadata:
                with open(args.metadata, 'r') as f:
                    metadata = json.load(f)
            
            # Import dataset
            dataset_id = easy_dataset.import_dataset(args.input, args.name, args.format)
            print(f"Created dataset '{args.name}' with ID: {dataset_id}")
            
        elif args.command == 'load':
            df = easy_dataset.load_dataset(args.id)
            if args.info:
                print(f"Dataset Info:")
                print(f"  Rows: {len(df)}")
                print(f"  Columns: {len(df.columns)}")
                print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
                print(f"  Column types:")
                for col, dtype in df.dtypes.items():
                    print(f"    {col}: {dtype}")
            else:
                print(f"Dataset {args.id} (first {args.head} rows):")
                print(df.head(args.head))
            
        elif args.command == 'validate':
            validation_rules = None
            if args.rules:
                with open(args.rules, 'r') as f:
                    validation_rules = json.load(f)
            
            results = easy_dataset.validate_dataset(args.id, validation_rules)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Validation results saved to {args.output}")
            else:
                print(f"Validation Results for {args.id}:")
                print(f"  Status: {'PASSED' if results['passed'] else 'FAILED'}")
                print(f"  Errors: {len(results['errors'])}")
                print(f"  Warnings: {len(results['warnings'])}")
                
                if results['errors']:
                    print("  Errors:")
                    for error in results['errors']:
                        print(f"    - {error}")
                
                if results['warnings']:
                    print("  Warnings:")
                    for warning in results['warnings']:
                        print(f"    - {warning}")
                
                print(f"  Stats: {results['stats']}")
            
        elif args.command == 'process':
            try:
                transformations = json.loads(args.transformations)
            except json.JSONDecodeError as e:
                print(f"Error parsing transformations JSON: {e}")
                sys.exit(1)
            
            new_dataset_id = easy_dataset.process_dataset(args.id, transformations)
            print(f"Processed dataset {args.id} -> {new_dataset_id}")
            
        elif args.command == 'visualize':
            viz_config = {
                'type': args.type,
                'save': args.save
            }
            
            if args.type == 'histogram':
                if not args.column:
                    print("Error: --column required for histogram visualization")
                    sys.exit(1)
                viz_config['column'] = args.column
            elif args.type in ['scatter', 'bar', 'line']:
                if not args.x or not args.y:
                    print(f"Error: --x and --y required for {args.type} visualization")
                    sys.exit(1)
                viz_config['x'] = args.x
                viz_config['y'] = args.y
            
            if args.output:
                viz_config['output_path'] = args.output
            
            viz_path = easy_dataset.visualize_dataset(args.id, viz_config)
            if viz_path:
                print(f"Visualization saved to: {viz_path}")
            else:
                print("Visualization created (not saved)")
            
        elif args.command == 'export':
            export_path = easy_dataset.export_dataset(args.id, args.format, args.output)
            print(f"Dataset {args.id} exported to: {export_path}")
            
        elif args.command == 'import':
            dataset_id = easy_dataset.import_dataset(args.input, args.name, args.format)
            print(f"Imported dataset '{args.name}' with ID: {dataset_id}")
            
        elif args.command == 'list':
            datasets = easy_dataset.list_datasets()
            
            if args.format == 'json':
                print(json.dumps(datasets, indent=2))
            else:
                if not datasets:
                    print("No datasets found")
                else:
                    print(f"Available datasets ({len(datasets)}):")
                    print(f"{'ID':<20} {'Name':<20} {'Rows':<10} {'Columns':<10} {'Size (MB)':<10}")
                    print("-" * 70)
                    for dataset in datasets:
                        metadata = dataset.get('metadata', {})
                        rows = metadata.get('rows', 'N/A')
                        columns = metadata.get('columns', 'N/A')
                        size = metadata.get('size_mb', 'N/A')
                        if isinstance(size, (int, float)):
                            size = f"{size:.2f}"
                        print(f"{dataset['id']:<20} {dataset['name']:<20} {rows:<10} {columns:<10} {size:<10}")
            
        elif args.command == 'delete':
            if not args.force:
                confirm = input(f"Are you sure you want to delete dataset {args.id}? (y/N): ")
                if confirm.lower() != 'y':
                    print("Deletion cancelled")
                    return
            
            success = easy_dataset.delete_dataset(args.id)
            if success:
                print(f"Dataset {args.id} deleted successfully")
            else:
                print(f"Failed to delete dataset {args.id}")
                sys.exit(1)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 