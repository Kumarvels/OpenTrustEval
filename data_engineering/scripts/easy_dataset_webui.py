#!/usr/bin/env python3
"""
Dataset Management WebUI
Gradio-based web interface for dataset management operations
"""

import os
import sys
import gradio as gr
import json
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from data_engineering.dataset_integration import DatasetManager

# Initialize dataset manager
dataset_manager = DatasetManager()

def create_dataset_from_upload(file, name, format_type):
    """Create dataset from uploaded file"""
    try:
        if file is None:
            return "Error: Please upload a file"
        
        # Save uploaded file to temporary location
        temp_path = tempfile.mktemp(suffix=os.path.splitext(file.name)[1])
        shutil.copy2(file.name, temp_path)
        
        # Import dataset
        dataset_id = dataset_manager.import_dataset(temp_path, name, format_type)
        
        # Clean up temp file
        os.unlink(temp_path)
        
        return f"Successfully created dataset '{name}' with ID: {dataset_id}"
    except Exception as e:
        return f"Error creating dataset: {e}"

def list_datasets():
    """List all datasets"""
    try:
        datasets = dataset_manager.list_datasets()
        if not datasets:
            return "No datasets found"
        
        result = "Available datasets:\n\n"
        for dataset in datasets:
            metadata = dataset.get('metadata', {})
            result += f"ID: {dataset['id']}\n"
            result += f"Name: {dataset['name']}\n"
            result += f"Rows: {metadata.get('rows', 'N/A')}\n"
            result += f"Columns: {metadata.get('columns', 'N/A')}\n"
            result += f"Created: {metadata.get('created_at', 'N/A')}\n"
            result += "-" * 40 + "\n"
        
        return result
    except Exception as e:
        return f"Error listing datasets: {e}"

def load_dataset(dataset_id):
    """Load and display dataset"""
    try:
        df = dataset_manager.load_dataset(dataset_id)
        return f"Dataset loaded successfully!\n\nShape: {df.shape}\n\nFirst 10 rows:\n{df.head(10).to_string()}"
    except Exception as e:
        return f"Error loading dataset: {e}"

def validate_dataset(dataset_id, validation_rules_json):
    """Validate dataset"""
    try:
        validation_rules = None
        if validation_rules_json.strip():
            validation_rules = json.loads(validation_rules_json)
        
        results = dataset_manager.validate_dataset(dataset_id, validation_rules)
        
        result = f"Validation: {'PASSED' if results['passed'] else 'FAILED'}\n\n"
        
        if results['errors']:
            result += "Errors:\n"
            for error in results['errors']:
                result += f"  - {error}\n"
            result += "\n"
        
        if results['warnings']:
            result += "Warnings:\n"
            for warning in results['warnings']:
                result += f"  - {warning}\n"
            result += "\n"
        
        result += f"Stats: {json.dumps(results['stats'], indent=2)}"
        
        return result
    except Exception as e:
        return f"Error validating dataset: {e}"

def process_dataset(dataset_id, transformations_json):
    """Process dataset with transformations"""
    try:
        transformations = json.loads(transformations_json)
        new_dataset_id = dataset_manager.process_dataset(dataset_id, transformations)
        return f"Successfully processed dataset {dataset_id} -> {new_dataset_id}"
    except Exception as e:
        return f"Error processing dataset: {e}"

def visualize_dataset(dataset_id, viz_type, x_col, y_col, column, save_viz):
    """Create visualization for dataset"""
    try:
        viz_config = {
            'type': viz_type,
            'save': save_viz
        }
        
        if viz_type == 'scatter':
            viz_config.update({'x': x_col, 'y': y_col})
        elif viz_type == 'histogram':
            viz_config.update({'column': column})
        elif viz_type in ['bar', 'line']:
            viz_config.update({'x': x_col, 'y': y_col})
        
        viz_path = dataset_manager.visualize_dataset(dataset_id, viz_config)
        
        if viz_path and save_viz:
            return f"Visualization saved to: {viz_path}"
        else:
            return "Visualization created successfully"
    except Exception as e:
        return f"Error creating visualization: {e}"

def export_dataset(dataset_id, format_type, output_path):
    """Export dataset"""
    try:
        export_path = dataset_manager.export_dataset(dataset_id, format_type, output_path)
        return f"Successfully exported dataset to: {export_path}"
    except Exception as e:
        return f"Error exporting dataset: {e}"

def delete_dataset(dataset_id):
    """Delete dataset"""
    try:
        success = dataset_manager.delete_dataset(dataset_id)
        if success:
            return f"Successfully deleted dataset {dataset_id}"
        else:
            return f"Failed to delete dataset {dataset_id}"
    except Exception as e:
        return f"Error deleting dataset: {e}"

def quality_filter_dataset(dataset_id, min_trust, features):
    try:
        features_list = [f.strip() for f in features.split(',')] if features else None
        new_dataset_id = dataset_manager.create_quality_filtered_dataset(dataset_id, min_trust, features_list)
        return f"Created quality-filtered dataset: {new_dataset_id}"
    except Exception as e:
        return f"Error in quality-based filtering: {e}"

def generate_quality_report(dataset_id, output_path):
    try:
        df = dataset_manager.load_dataset(dataset_id)
        
        # Use fallback quality manager for reporting
        from data_engineering.cleanlab_integration import FallbackDataQualityManager
        quality_manager = FallbackDataQualityManager()
        report = quality_manager.generate_quality_report(df, output_path)
        
        if output_path:
            return f"Quality report saved to: {output_path}"
        else:
            return report
    except Exception as e:
        return f"Error generating quality report: {e}"

# --- Cleanlab Datalab Plugin Integration ---
def run_cleanlab_on_dataset(dataset_id, label_column, save_output):
    try:
        df = dataset_manager.load_dataset(dataset_id)
        labels = df[label_column].tolist() if label_column and label_column in df.columns else None
        # Use DataLifecycleManager for plugin logic
        from data_engineering.data_lifecycle import DataLifecycleManager
        config = {'enable_cleanlab': True, 'cleanlab_label_column': label_column}
        manager = DataLifecycleManager(config=config)
        result = manager.run_cleanlab_plugin(df, labels=labels)
        if save_output and result:
            import json
            with open('cleanlab_issues_webui.json', 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            return f"Output saved to cleanlab_issues_webui.json\n\n{result}"
        return result
    except Exception as e:
        return f"Error running Cleanlab plugin: {e}"

# Create Gradio interface
with gr.Blocks(title="Dataset Management WebUI") as demo:
    gr.Markdown("# Dataset Management WebUI")
    
    with gr.Tab("Create Dataset"):
        gr.Markdown("## Upload and Create Dataset")
        with gr.Row():
            file_input = gr.File(label="Upload File")
            name_input = gr.Textbox(label="Dataset Name", placeholder="Enter dataset name")
            format_dropdown = gr.Dropdown(
                choices=["csv", "json", "parquet", "excel"],
                label="File Format",
                value="csv"
            )
        create_btn = gr.Button("Create Dataset")
        create_output = gr.Textbox(label="Result", lines=3)
        create_btn.click(
            create_dataset_from_upload,
            inputs=[file_input, name_input, format_dropdown],
            outputs=create_output
        )
    
    with gr.Tab("List Datasets"):
        gr.Markdown("## List All Datasets")
        list_btn = gr.Button("List Datasets")
        list_output = gr.Textbox(label="Datasets", lines=15)
        list_btn.click(list_datasets, outputs=list_output)
    
    with gr.Tab("Load Dataset"):
        gr.Markdown("## Load and View Dataset")
        load_id = gr.Textbox(label="Dataset ID", placeholder="Enter dataset ID")
        load_btn = gr.Button("Load Dataset")
        load_output = gr.Textbox(label="Dataset Content", lines=15)
        load_btn.click(load_dataset, inputs=load_id, outputs=load_output)
    
    with gr.Tab("Validate Dataset"):
        gr.Markdown("## Validate Dataset")
        validate_id = gr.Textbox(label="Dataset ID", placeholder="Enter dataset ID")
        validation_rules = gr.Textbox(
            label="Validation Rules (JSON, optional)",
            placeholder='{"age_range": "lambda df: (df[\"age\"] >= 0) & (df[\"age\"] <= 120)"}',
            lines=3
        )
        validate_btn = gr.Button("Validate Dataset")
        validate_output = gr.Textbox(label="Validation Results", lines=10)
        validate_btn.click(
            validate_dataset,
            inputs=[validate_id, validation_rules],
            outputs=validate_output
        )
    
    with gr.Tab("Process Dataset"):
        gr.Markdown("## Process Dataset with Transformations")
        process_id = gr.Textbox(label="Dataset ID", placeholder="Enter dataset ID")
        transformations = gr.Textbox(
            label="Transformations (JSON)",
            placeholder='[{"operation": "filter", "params": {"condition": "age > 30"}}]',
            lines=5
        )
        process_btn = gr.Button("Process Dataset")
        process_output = gr.Textbox(label="Result", lines=3)
        process_btn.click(
            process_dataset,
            inputs=[process_id, transformations],
            outputs=process_output
        )
    
    with gr.Tab("Visualize Dataset"):
        gr.Markdown("## Create Visualizations")
        viz_id = gr.Textbox(label="Dataset ID", placeholder="Enter dataset ID")
        with gr.Row():
            viz_type = gr.Dropdown(
                choices=["scatter", "histogram", "bar", "line", "correlation"],
                label="Visualization Type",
                value="scatter"
            )
            save_viz = gr.Checkbox(label="Save Visualization", value=True)
        
        with gr.Row():
            x_col = gr.Textbox(label="X Column", placeholder="Column name for X-axis")
            y_col = gr.Textbox(label="Y Column", placeholder="Column name for Y-axis")
            column = gr.Textbox(label="Column (for histogram)", placeholder="Column name")
        
        viz_btn = gr.Button("Create Visualization")
        viz_output = gr.Textbox(label="Result", lines=3)
        viz_btn.click(
            visualize_dataset,
            inputs=[viz_id, viz_type, x_col, y_col, column, save_viz],
            outputs=viz_output
        )
    
    with gr.Tab("Export Dataset"):
        gr.Markdown("## Export Dataset")
        export_id = gr.Textbox(label="Dataset ID", placeholder="Enter dataset ID")
        with gr.Row():
            export_format = gr.Dropdown(
                choices=["csv", "json", "parquet", "excel"],
                label="Export Format",
                value="csv"
            )
            export_path = gr.Textbox(label="Output Path (optional)", placeholder="Leave empty for auto-generated path")
        export_btn = gr.Button("Export Dataset")
        export_output = gr.Textbox(label="Result", lines=3)
        export_btn.click(
            export_dataset,
            inputs=[export_id, export_format, export_path],
            outputs=export_output
        )
    
    with gr.Tab("Delete Dataset"):
        gr.Markdown("## Delete Dataset")
        delete_id = gr.Textbox(label="Dataset ID", placeholder="Enter dataset ID")
        delete_btn = gr.Button("Delete Dataset", variant="stop")
        delete_output = gr.Textbox(label="Result", lines=3)
        delete_btn.click(delete_dataset, inputs=delete_id, outputs=delete_output)

    with gr.Tab("Quality-Based Filtering (Cleanlab)"):
        gr.Markdown("## Filter Dataset by Cleanlab Trust Score")
        qf_id = gr.Textbox(label="Dataset ID", placeholder="Enter dataset ID")
        qf_min_trust = gr.Number(label="Min Trust Score", value=0.7)
        qf_features = gr.Textbox(label="Feature Columns (comma-separated, optional)", placeholder="e.g., age,salary")
        qf_btn = gr.Button("Filter Dataset")
        qf_output = gr.Textbox(label="Result", lines=3)
        qf_btn.click(quality_filter_dataset, inputs=[qf_id, qf_min_trust, qf_features], outputs=qf_output)

    with gr.Tab("Quality Report (Cleanlab)"):
        gr.Markdown("## Generate Cleanlab Quality Report")
        qr_id = gr.Textbox(label="Dataset ID", placeholder="Enter dataset ID")
        qr_output_path = gr.Textbox(label="Output Path (optional)", placeholder="Leave empty for JSON output")
        qr_btn = gr.Button("Generate Report")
        qr_output = gr.Textbox(label="Report / Result", lines=10)
        qr_btn.click(generate_quality_report, inputs=[qr_id, qr_output_path], outputs=qr_output)

    with gr.Tab("Data Issue Detection (Cleanlab)"):
        gr.Markdown("## Run Cleanlab Datalab Plugin on Dataset\n- Select a dataset and (optionally) label column.\n- Requires Cleanlab to be installed and licensed.")
        cleanlab_id = gr.Textbox(label="Dataset ID", placeholder="Enter dataset ID")
        cleanlab_label = gr.Textbox(label="Label Column (optional)", placeholder="e.g. job_title")
        cleanlab_save = gr.Checkbox(label="Save output to file", value=True)
        cleanlab_btn = gr.Button("Run Cleanlab Plugin")
        cleanlab_output = gr.Textbox(label="Plugin Output", lines=10)
        cleanlab_btn.click(run_cleanlab_on_dataset, inputs=[cleanlab_id, cleanlab_label, cleanlab_save], outputs=cleanlab_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861) 