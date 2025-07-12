#!/usr/bin/env python3
"""
Dataset Management CLI
Command-line interface for dataset management operations

Usage Examples:
python dataset_cli.py create --name "my_dataset" --input data.csv --format csv
python dataset_cli.py validate --id dataset_123
python dataset_cli.py process --id dataset_123 --transformations '[{"operation": "filter", "params": {"condition": "age > 30"}}]'
python dataset_cli.py visualize --id dataset_123 --type scatter --x age --y salary --save
python dataset_cli.py export --id dataset_123 --format json --output data.json
python dataset_cli.py import --name "new_dataset" --input data.json --format json
python dataset_cli.py list
python dataset_cli.py delete --id dataset_123
"""

import os
import sys
import argparse
import json
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from data_engineering.dataset_integration import DatasetManager

def parse_json_config(config_str):
    """Parse JSON config string with better error handling."""
    try:
        return json.loads(config_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON config: {e}")
        print("Make sure to use proper JSON format with double quotes.")
        print("Example: '{\"operation\": \"filter\", \"params\": {\"condition\": \"age > 30\"}}'")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Dataset Management CLI")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Global arguments
    parser.add_argument('--base-path', type=str, default="./datasets", 
                       help='Base path for dataset storage')
    
    # Create dataset
    create_parser = subparsers.add_parser('create', help='Create a new dataset')
    create_parser.add_argument('--name', required=True, help='Dataset name')
    create_parser.add_argument('--input', required=True, help='Input file path')
    create_parser.add_argument('--format', default='csv', 
                              choices=['csv', 'json', 'parquet', 'excel'], 
                              help='Input file format')
    create_parser.add_argument('--schema', help='Schema file path (JSON)')
    
    # Load dataset
    load_parser = subparsers.add_parser('load', help='Load a dataset')
    load_parser.add_argument('--id', required=True, help='Dataset ID')
    load_parser.add_argument('--output', help='Output file path (optional)')
    
    # Validate dataset
    validate_parser = subparsers.add_parser('validate', help='Validate a dataset')
    validate_parser.add_argument('--id', required=True, help='Dataset ID')
    validate_parser.add_argument('--rules', help='Validation rules file (JSON)')
    validate_parser.add_argument('--output', help='Output results file (JSON)')
    
    # Process dataset
    process_parser = subparsers.add_parser('process', help='Process a dataset')
    process_parser.add_argument('--id', required=True, help='Dataset ID')
    process_parser.add_argument('--transformations', required=True, 
                               help='Transformations as JSON string')
    
    # Visualize dataset
    visualize_parser = subparsers.add_parser('visualize', help='Visualize a dataset')
    visualize_parser.add_argument('--id', required=True, help='Dataset ID')
    visualize_parser.add_argument('--type', required=True, 
                                 choices=['scatter', 'histogram', 'bar', 'line', 'correlation'],
                                 help='Visualization type')
    visualize_parser.add_argument('--x', help='X-axis column')
    visualize_parser.add_argument('--y', help='Y-axis column')
    visualize_parser.add_argument('--column', help='Column for histogram')
    visualize_parser.add_argument('--save', action='store_true', help='Save visualization')
    
    # Export dataset
    export_parser = subparsers.add_parser('export', help='Export a dataset')
    export_parser.add_argument('--id', required=True, help='Dataset ID')
    export_parser.add_argument('--format', required=True, 
                              choices=['csv', 'json', 'parquet', 'excel'],
                              help='Export format')
    export_parser.add_argument('--output', help='Output file path')
    
    # Import dataset
    import_parser = subparsers.add_parser('import', help='Import a dataset')
    import_parser.add_argument('--name', required=True, help='Dataset name')
    import_parser.add_argument('--input', required=True, help='Input file path')
    import_parser.add_argument('--format', 
                              choices=['csv', 'json', 'parquet', 'excel'],
                              help='Input file format (auto-detected if not specified)')
    
    # List datasets
    list_parser = subparsers.add_parser('list', help='List all datasets')
    list_parser.add_argument('--format', choices=['table', 'json'], default='table',
                            help='Output format')
    
    # Delete dataset
    delete_parser = subparsers.add_parser('delete', help='Delete a dataset')
    delete_parser.add_argument('--id', required=True, help='Dataset ID')
    delete_parser.add_argument('--force', action='store_true', help='Force deletion without confirmation')
    
    # Quality-based filtering (Cleanlab)
    quality_filter_parser = subparsers.add_parser('quality-filter', help='Filter dataset by Cleanlab trust score')
    quality_filter_parser.add_argument('--id', required=True, help='Dataset ID')
    quality_filter_parser.add_argument('--min-trust', type=float, default=0.7, help='Minimum trust score threshold')
    quality_filter_parser.add_argument('--features', help='Feature columns (comma-separated, optional)')

    # Quality report (Cleanlab)
    quality_report_parser = subparsers.add_parser('quality-report', help='Generate Cleanlab quality report')
    quality_report_parser.add_argument('--id', required=True, help='Dataset ID')
    quality_report_parser.add_argument('--output', help='Output file path (optional)')
    
    args = parser.parse_args()
    
    # Initialize dataset manager
    dataset_manager = DatasetManager(args.base_path)
    
    try:
        if args.command == 'create':
            # Load schema if provided
            schema = None
            if args.schema:
                with open(args.schema, 'r') as f:
                    schema = json.load(f)
            
            # Import dataset from file
            dataset_id = dataset_manager.import_dataset(args.input, args.name, args.format)
            print(f"Created dataset '{args.name}' with ID: {dataset_id}")
            
        elif args.command == 'load':
            df = dataset_manager.load_dataset(args.id)
            print(f"Loaded dataset {args.id} with {len(df)} rows and {len(df.columns)} columns")
            print("\nFirst 5 rows:")
            print(df.head())
            
            if args.output:
                df.to_csv(args.output, index=False)
                print(f"Saved to {args.output}")
                
        elif args.command == 'validate':
            # Load validation rules if provided
            validation_rules = None
            if args.rules:
                with open(args.rules, 'r') as f:
                    validation_rules = json.load(f)
            
            results = dataset_manager.validate_dataset(args.id, validation_rules)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Validation results saved to {args.output}")
            else:
                print(f"Validation: {'PASSED' if results['passed'] else 'FAILED'}")
                if results['errors']:
                    print("Errors:")
                    for error in results['errors']:
                        print(f"  - {error}")
                if results['warnings']:
                    print("Warnings:")
                    for warning in results['warnings']:
                        print(f"  - {warning}")
                print(f"Stats: {results['stats']}")
                
        elif args.command == 'process':
            transformations = parse_json_config(args.transformations)
            new_dataset_id = dataset_manager.process_dataset(args.id, transformations)
            print(f"Processed dataset {args.id} -> {new_dataset_id}")
            
        elif args.command == 'visualize':
            viz_config = {
                'type': args.type,
                'save': args.save
            }
            
            if args.type == 'scatter':
                viz_config.update({'x': args.x, 'y': args.y})
            elif args.type == 'histogram':
                viz_config.update({'column': args.column})
            elif args.type in ['bar', 'line']:
                viz_config.update({'x': args.x, 'y': args.y})
            
            viz_path = dataset_manager.visualize_dataset(args.id, viz_config)
            if viz_path:
                print(f"Visualization saved to: {viz_path}")
            else:
                print("Visualization created (not saved)")
                
        elif args.command == 'export':
            export_path = dataset_manager.export_dataset(args.id, args.format, args.output)
            print(f"Exported dataset {args.id} to: {export_path}")
            
        elif args.command == 'import':
            dataset_id = dataset_manager.import_dataset(args.input, args.name, args.format)
            print(f"Imported dataset '{args.name}' with ID: {dataset_id}")
            
        elif args.command == 'list':
            datasets = dataset_manager.list_datasets()
            
            if args.format == 'json':
                print(json.dumps(datasets, indent=2))
            else:
                print(f"Found {len(datasets)} datasets:")
                for dataset in datasets:
                    metadata = dataset.get('metadata', {})
                    print(f"  {dataset['id']}: {dataset['name']}")
                    print(f"    Path: {dataset['path']}")
                    print(f"    Rows: {metadata.get('rows', 'N/A')}")
                    print(f"    Columns: {metadata.get('columns', 'N/A')}")
                    print(f"    Created: {metadata.get('created_at', 'N/A')}")
                    print()
                    
        elif args.command == 'delete':
            if not args.force:
                confirm = input(f"Are you sure you want to delete dataset {args.id}? (y/N): ")
                if confirm.lower() != 'y':
                    print("Deletion cancelled")
                    return
            
            success = dataset_manager.delete_dataset(args.id)
            if success:
                print(f"Deleted dataset {args.id}")
            else:
                print(f"Failed to delete dataset {args.id}")
                
        elif args.command == 'quality-filter':
            features = args.features.split(',') if args.features else None
            new_dataset_id = dataset_manager.create_quality_filtered_dataset(args.id, args.min_trust, features)
            print(f"Created quality-filtered dataset: {new_dataset_id}")

        elif args.command == 'quality-report':
            # Quality report generation
            try:
                df = dataset_manager.load_dataset(args.id)
                
                # Use fallback quality manager for reporting
                from data_engineering.cleanlab_integration import FallbackDataQualityManager
                quality_manager = FallbackDataQualityManager()
                report = quality_manager.generate_quality_report(df, args.output)
                
                if args.output:
                    print(f"Quality report saved to: {args.output}")
                else:
                    print(report)
            except Exception as e:
                print(f"Error generating quality report: {e}")
                
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 