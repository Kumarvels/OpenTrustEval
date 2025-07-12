#!/usr/bin/env python3
"""
Easy Dataset WebUI - Gradio-based web interface for Easy Dataset management
Provides user-friendly access to all Easy Dataset features
"""

import os
import sys
import gradio as gr
import json
import pandas as pd
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from data_engineering.easy_dataset_integration import EasyDatasetManager

# Initialize Easy Dataset manager
easy_dataset = EasyDatasetManager()

def create_dataset_from_upload(name, file, format_type, schema_text, metadata_text):
    """Create dataset from uploaded file"""
    try:
        if not name or not file:
            return "Error: Name and file are required"
        
        # Save uploaded file temporarily
        temp_path = tempfile.mktemp(suffix=f".{format_type}")
        with open(temp_path, 'wb') as f:
            f.write(file.read())
        
        # Parse schema and metadata if provided
        schema = None
        if schema_text.strip():
            try:
                schema = json.loads(schema_text)
            except json.JSONDecodeError:
                return "Error: Invalid JSON in schema"
        
        metadata = None
        if metadata_text.strip():
            try:
                metadata = json.loads(metadata_text)
            except json.JSONDecodeError:
                return "Error: Invalid JSON in metadata"
        
        # Create dataset
        dataset_id = easy_dataset.import_dataset(temp_path, name, format_type)
        
        # Clean up temp file
        os.unlink(temp_path)
        
        return f"Successfully created dataset '{name}' with ID: {dataset_id}"
    
    except Exception as e:
        return f"Error creating dataset: {str(e)}"

def list_datasets():
    """List all available datasets"""
    try:
        datasets = easy_dataset.list_datasets()
        if not datasets:
            return "No datasets found"
        
        result = "Available Datasets:\n\n"
        for dataset in datasets:
            metadata = dataset.get('metadata', {})
            result += f"ID: {dataset['id']}\n"
            result += f"Name: {dataset['name']}\n"
            result += f"Rows: {metadata.get('rows', 'N/A')}\n"
            result += f"Columns: {metadata.get('columns', 'N/A')}\n"
            result += f"Size: {metadata.get('size_mb', 'N/A')} MB\n"
            result += "-" * 40 + "\n"
        
        return result
    
    except Exception as e:
        return f"Error listing datasets: {str(e)}"

def load_dataset_info(dataset_id):
    """Load and display dataset information"""
    try:
        if not dataset_id:
            return "Please enter a dataset ID"
        
        df = easy_dataset.load_dataset(dataset_id)
        
        info = f"Dataset: {dataset_id}\n"
        info += f"Rows: {len(df)}\n"
        info += f"Columns: {len(df.columns)}\n"
        info += f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB\n\n"
        info += "Column types:\n"
        for col, dtype in df.dtypes.items():
            info += f"  {col}: {dtype}\n"
        info += f"\nFirst 5 rows:\n{df.head().to_string()}"
        
        return info
    
    except Exception as e:
        return f"Error loading dataset: {str(e)}"

def validate_dataset(dataset_id, validation_rules_text):
    """Validate a dataset"""
    try:
        if not dataset_id:
            return "Please enter a dataset ID"
        
        validation_rules = None
        if validation_rules_text.strip():
            try:
                validation_rules = json.loads(validation_rules_text)
            except json.JSONDecodeError:
                return "Error: Invalid JSON in validation rules"
        
        results = easy_dataset.validate_dataset(dataset_id, validation_rules)
        
        result = f"Validation Results for {dataset_id}:\n"
        result += f"Status: {'PASSED' if results['passed'] else 'FAILED'}\n"
        result += f"Errors: {len(results['errors'])}\n"
        result += f"Warnings: {len(results['warnings'])}\n\n"
        
        if results['errors']:
            result += "Errors:\n"
            for error in results['errors']:
                result += f"  - {error}\n"
        
        if results['warnings']:
            result += "Warnings:\n"
            for warning in results['warnings']:
                result += f"  - {warning}\n"
        
        result += f"\nStats: {json.dumps(results['stats'], indent=2)}"
        
        return result
    
    except Exception as e:
        return f"Error validating dataset: {str(e)}"

def process_dataset(dataset_id, transformations_text):
    """Process a dataset with transformations"""
    try:
        if not dataset_id or not transformations_text:
            return "Please enter dataset ID and transformations"
        
        try:
            transformations = json.loads(transformations_text)
        except json.JSONDecodeError:
            return "Error: Invalid JSON in transformations"
        
        new_dataset_id = easy_dataset.process_dataset(dataset_id, transformations)
        return f"Successfully processed dataset {dataset_id} -> {new_dataset_id}"
    
    except Exception as e:
        return f"Error processing dataset: {str(e)}"

def visualize_dataset(dataset_id, viz_type, x_column, y_column, column, save_viz):
    """Create visualization for dataset"""
    try:
        if not dataset_id:
            return "Please enter a dataset ID"
        
        viz_config = {
            'type': viz_type,
            'save': save_viz
        }
        
        if viz_type == 'histogram':
            if not column:
                return "Error: Column required for histogram"
            viz_config['column'] = column
        elif viz_type in ['scatter', 'bar', 'line']:
            if not x_column or not y_column:
                return f"Error: X and Y columns required for {viz_type}"
            viz_config['x'] = x_column
            viz_config['y'] = y_column
        
        viz_path = easy_dataset.visualize_dataset(dataset_id, viz_config)
        
        if viz_path:
            return f"Visualization saved to: {viz_path}"
        else:
            return "Visualization created (not saved)"
    
    except Exception as e:
        return f"Error creating visualization: {str(e)}"

def export_dataset(dataset_id, format_type, output_path):
    """Export dataset"""
    try:
        if not dataset_id:
            return "Please enter a dataset ID"
        
        export_path = easy_dataset.export_dataset(dataset_id, format_type, output_path)
        return f"Dataset {dataset_id} exported to: {export_path}"
    
    except Exception as e:
        return f"Error exporting dataset: {str(e)}"

def delete_dataset(dataset_id):
    """Delete a dataset"""
    try:
        if not dataset_id:
            return "Please enter a dataset ID"
        
        success = easy_dataset.delete_dataset(dataset_id)
        if success:
            return f"Dataset {dataset_id} deleted successfully"
        else:
            return f"Failed to delete dataset {dataset_id}"
    
    except Exception as e:
        return f"Error deleting dataset: {str(e)}"

def get_transformation_examples():
    """Get example transformations"""
    examples = {
        "Filter rows": '[{"operation": "filter", "params": {"condition": "age > 30"}}]',
        "Sort by column": '[{"operation": "sort", "params": {"columns": ["salary"], "ascending": false}}]',
        "Drop columns": '[{"operation": "drop_columns", "params": {"columns": ["id", "temp_column"]}}]',
        "Rename columns": '[{"operation": "rename_columns", "params": {"mapping": {"old_name": "new_name"}}}]',
        "Multiple operations": '''[
            {"operation": "filter", "params": {"condition": "age > 25"}},
            {"operation": "sort", "params": {"columns": ["salary"], "ascending": false}},
            {"operation": "drop_columns", "params": {"columns": ["temp_column"]}}
        ]'''
    }
    return json.dumps(examples, indent=2)

def get_validation_examples():
    """Get example validation rules"""
    examples = {
        "Check for nulls": "lambda df: df.isnull().sum().sum() == 0",
        "Check data types": "lambda df: df['age'].dtype == 'int64'",
        "Check value ranges": "lambda df: (df['age'] >= 0) & (df['age'] <= 120)",
        "Check unique values": "lambda df: df['id'].is_unique"
    }
    return json.dumps(examples, indent=2)

# Create Gradio interface
with gr.Blocks(title="Easy Dataset Manager") as demo:
    gr.Markdown("# 🗂️ Easy Dataset Manager")
    gr.Markdown("Manage datasets with validation, processing, visualization, and export/import capabilities")
    
    with gr.Tab("📁 Create Dataset"):
        gr.Markdown("### Create a new dataset from uploaded file")
        with gr.Row():
            with gr.Column():
                name_input = gr.Textbox(label="Dataset Name", placeholder="Enter dataset name")
                file_input = gr.File(label="Upload File")
                format_dropdown = gr.Dropdown(
                    choices=["csv", "json", "parquet", "excel"],
                    value="csv",
                    label="File Format"
                )
                create_btn = gr.Button("Create Dataset", variant="primary")
                create_output = gr.Textbox(label="Result", lines=5)
            
            with gr.Column():
                gr.Markdown("### Optional Schema (JSON)")
                schema_input = gr.Textbox(
                    label="Schema",
                    placeholder='{"column_name": {"type": "string", "nullable": false}}',
                    lines=5
                )
                gr.Markdown("### Optional Metadata (JSON)")
                metadata_input = gr.Textbox(
                    label="Metadata",
                    placeholder='{"description": "My dataset", "source": "API"}',
                    lines=5
                )
        
        create_btn.click(
            create_dataset_from_upload,
            inputs=[name_input, file_input, format_dropdown, schema_input, metadata_input],
            outputs=create_output
        )
    
    with gr.Tab("📋 List Datasets"):
        gr.Markdown("### View all available datasets")
        list_btn = gr.Button("List Datasets", variant="primary")
        list_output = gr.Textbox(label="Datasets", lines=15)
        list_btn.click(list_datasets, outputs=list_output)
    
    with gr.Tab("🔍 Load Dataset"):
        gr.Markdown("### Load and view dataset information")
        load_id_input = gr.Textbox(label="Dataset ID", placeholder="Enter dataset ID")
        load_btn = gr.Button("Load Dataset", variant="primary")
        load_output = gr.Textbox(label="Dataset Info", lines=15)
        load_btn.click(load_dataset_info, inputs=load_id_input, outputs=load_output)
    
    with gr.Tab("✅ Validate Dataset"):
        gr.Markdown("### Validate dataset against schema and rules")
        with gr.Row():
            with gr.Column():
                validate_id_input = gr.Textbox(label="Dataset ID", placeholder="Enter dataset ID")
                validate_btn = gr.Button("Validate Dataset", variant="primary")
                validate_output = gr.Textbox(label="Validation Results", lines=15)
            
            with gr.Column():
                gr.Markdown("### Custom Validation Rules (JSON)")
                validation_rules_input = gr.Textbox(
                    label="Validation Rules",
                    placeholder='{"rule_name": "lambda df: condition"}',
                    lines=10
                )
                gr.Markdown("### Example Rules")
                validation_examples_btn = gr.Button("Show Examples")
                validation_examples_output = gr.Textbox(label="Examples", lines=8)
        
        validate_btn.click(
            validate_dataset,
            inputs=[validate_id_input, validation_rules_input],
            outputs=validate_output
        )
        validation_examples_btn.click(get_validation_examples, outputs=validation_examples_output)
    
    with gr.Tab("⚙️ Process Dataset"):
        gr.Markdown("### Apply transformations to dataset")
        with gr.Row():
            with gr.Column():
                process_id_input = gr.Textbox(label="Dataset ID", placeholder="Enter dataset ID")
                process_btn = gr.Button("Process Dataset", variant="primary")
                process_output = gr.Textbox(label="Result", lines=5)
            
            with gr.Column():
                gr.Markdown("### Transformations (JSON)")
                transformations_input = gr.Textbox(
                    label="Transformations",
                    placeholder='[{"operation": "filter", "params": {"condition": "age > 30"}}]',
                    lines=10
                )
                gr.Markdown("### Example Transformations")
                transform_examples_btn = gr.Button("Show Examples")
                transform_examples_output = gr.Textbox(label="Examples", lines=8)
        
        process_btn.click(
            process_dataset,
            inputs=[process_id_input, transformations_input],
            outputs=process_output
        )
        transform_examples_btn.click(get_transformation_examples, outputs=transform_examples_output)
    
    with gr.Tab("📊 Visualize Dataset"):
        gr.Markdown("### Create visualizations for dataset")
        with gr.Row():
            with gr.Column():
                viz_id_input = gr.Textbox(label="Dataset ID", placeholder="Enter dataset ID")
                viz_type_dropdown = gr.Dropdown(
                    choices=["histogram", "scatter", "bar", "line", "correlation"],
                    value="scatter",
                    label="Visualization Type"
                )
                x_column_input = gr.Textbox(label="X Column", placeholder="Column name for X-axis")
                y_column_input = gr.Textbox(label="Y Column", placeholder="Column name for Y-axis")
                column_input = gr.Textbox(label="Column (for histogram)", placeholder="Column name")
                save_viz_checkbox = gr.Checkbox(label="Save visualization", value=True)
                viz_btn = gr.Button("Create Visualization", variant="primary")
                viz_output = gr.Textbox(label="Result", lines=5)
    
        viz_btn.click(
            visualize_dataset,
            inputs=[viz_id_input, viz_type_dropdown, x_column_input, y_column_input, column_input, save_viz_checkbox],
            outputs=viz_output
        )
    
    with gr.Tab("📤 Export Dataset"):
        gr.Markdown("### Export dataset to various formats")
        export_id_input = gr.Textbox(label="Dataset ID", placeholder="Enter dataset ID")
        export_format_dropdown = gr.Dropdown(
            choices=["csv", "json", "parquet", "excel"],
            value="csv",
            label="Export Format"
        )
        export_path_input = gr.Textbox(label="Output Path (optional)", placeholder="Leave empty for auto-generated path")
        export_btn = gr.Button("Export Dataset", variant="primary")
        export_output = gr.Textbox(label="Result", lines=5)
        
        export_btn.click(
            export_dataset,
            inputs=[export_id_input, export_format_dropdown, export_path_input],
            outputs=export_output
        )
    
    with gr.Tab("🗑️ Delete Dataset"):
        gr.Markdown("### Delete a dataset")
        delete_id_input = gr.Textbox(label="Dataset ID", placeholder="Enter dataset ID")
        delete_btn = gr.Button("Delete Dataset", variant="stop")
        delete_output = gr.Textbox(label="Result", lines=5)
        
        delete_btn.click(delete_dataset, inputs=delete_id_input, outputs=delete_output)

# Launch the interface
if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7861) 