#!/usr/bin/env python3
"""
LLM Model Manager & Tuning WebUI
Gradio-based web interface for LLM model management and tuning
"""

import os
import sys
import gradio as gr
import json
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from llm_engineering.llm_lifecycle import LLMLifecycleManager
    LLM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: LLM engineering not available: {e}")
    LLM_AVAILABLE = False

from llm_engineering.workflows import ecommerce_support
from llm_engineering.workflows.logging import get_workflow_logs

# Initialize manager with error handling
manager = None
if LLM_AVAILABLE:
    try:
        manager = LLMLifecycleManager()
        # Add some default models for demo
        if not manager.list_models():
            manager.add_model(
                "demo_llama", 
                "llama_factory", 
                {"model_name": "Llama-3-8B", "model_path": None},
                persist=False
            )
    except Exception as e:
        print(f"Warning: Failed to initialize LLM manager: {e}")
        manager = None

# --- Model Management Functions ---
def list_models():
    """List all models"""
    if not manager:
        return "LLM Manager not available. Check dependencies and configuration."
    
    try:
        models = manager.list_models()
        if not models:
            return "No models found. Add models using the 'Add Model' tab."
        return "\n".join(models)
    except Exception as e:
        return f"Error listing models: {e}"

def add_model(name, provider_type, config_json):
    """Add a new model"""
    if not manager:
        return "LLM Manager not available. Check dependencies and configuration."
    
    try:
        if not name or not provider_type:
            return "Error: Model name and provider type are required."
        
        config = {}
        if config_json.strip():
            try:
                config = json.loads(config_json)
            except json.JSONDecodeError:
                return "Error: Invalid JSON configuration."
        
        manager.add_model(name, provider_type, config, persist=False)
        return f"Successfully added model '{name}' of type '{provider_type}'."
    except Exception as e:
        return f"Error adding model: {e}"

def remove_model(name):
    """Remove a model"""
    if not manager:
        return "LLM Manager not available. Check dependencies and configuration."
    
    try:
        if not name:
            return "Error: Model name is required."
        
        manager.remove_model(name, persist=False)
        return f"Successfully removed model '{name}'."
    except Exception as e:
        return f"Error removing model: {e}"

def update_model(name, config_json):
    """Update model configuration"""
    if not manager:
        return "LLM Manager not available. Check dependencies and configuration."
    
    try:
        if not name:
            return "Error: Model name is required."
        
        config = {}
        if config_json.strip():
            try:
                config = json.loads(config_json)
            except json.JSONDecodeError:
                return "Error: Invalid JSON configuration."
        
        manager.update_model(name, config, persist=False)
        return f"Successfully updated model '{name}'."
    except Exception as e:
        return f"Error updating model: {e}"

# --- Tuning Functions ---
def fine_tune_model(name, dataset_path, extra_kwargs_json):
    """Fine-tune a model"""
    if not manager:
        return "LLM Manager not available. Check dependencies and configuration."
    
    try:
        if not name:
            return "Error: Model name is required."
        
        provider = manager.llm_providers.get(name)
        if provider is None:
            return f"Error: Provider '{name}' not found. Available providers: {list(manager.llm_providers.keys())}"
        
        extra_kwargs = {}
        if extra_kwargs_json.strip():
            try:
                extra_kwargs = json.loads(extra_kwargs_json)
            except json.JSONDecodeError:
                return "Error: Invalid JSON in extra kwargs."
        
        # For demo purposes, simulate fine-tuning
        if not dataset_path:
            dataset_path = "demo_dataset.csv"
        
        result = f"Fine-tuning simulation for model '{name}' with dataset '{dataset_path}'"
        if extra_kwargs:
            result += f"\nExtra kwargs: {extra_kwargs}"
        
        return result
    except Exception as e:
        return f"Error during fine-tuning: {e}"

def evaluate_model(name, dataset_path, extra_kwargs_json):
    """Evaluate a model"""
    if not manager:
        return "LLM Manager not available. Check dependencies and configuration."
    
    try:
        if not name:
            return "Error: Model name is required."
        
        provider = manager.llm_providers.get(name)
        if provider is None:
            return f"Error: Provider '{name}' not found. Available providers: {list(manager.llm_providers.keys())}"
        
        extra_kwargs = {}
        if extra_kwargs_json.strip():
            try:
                extra_kwargs = json.loads(extra_kwargs_json)
            except json.JSONDecodeError:
                return "Error: Invalid JSON in extra kwargs."
        
        # For demo purposes, simulate evaluation
        if not dataset_path:
            dataset_path = "demo_dataset.csv"
        
        result = f"Evaluation simulation for model '{name}' with dataset '{dataset_path}'"
        if extra_kwargs:
            result += f"\nExtra kwargs: {extra_kwargs}"
        
        return result
    except Exception as e:
        return f"Error during evaluation: {e}"

# --- Workflow Execution Functions ---
def run_order_status_workflow_ui(customer_id, order_id, status, expected_delivery):
    if not manager:
        return "LLM Manager not available."
    orchestrator = manager.llm_providers.get('langgraph_orchestrator')
    if not orchestrator:
        return "LangGraph orchestrator provider not found in config."
    try:
        result = ecommerce_support.run_order_status_workflow(
            orchestrator, customer_id, order_id, status, expected_delivery
        )
        return str(result)
    except Exception as e:
        return f"Workflow execution failed: {e}"

def run_returns_workflow_ui(customer_id, order_id, reason):
    if not manager:
        return "LLM Manager not available."
    orchestrator = manager.llm_providers.get('langgraph_orchestrator')
    if not orchestrator:
        return "LangGraph orchestrator provider not found in config."
    try:
        result = ecommerce_support.run_returns_workflow(
            orchestrator, customer_id, order_id, reason
        )
        return str(result)
    except Exception as e:
        return f"Workflow execution failed: {e}"

def run_escalation_workflow_ui(customer_id, issue):
    if not manager:
        return "LLM Manager not available."
    orchestrator = manager.llm_providers.get('langgraph_orchestrator')
    if not orchestrator:
        return "LangGraph orchestrator provider not found in config."
    try:
        result = ecommerce_support.run_escalation_workflow(
            orchestrator, customer_id, issue
        )
        return str(result)
    except Exception as e:
        return f"Workflow execution failed: {e}"

# --- Gradio UI ---
with gr.Blocks(title="LLM Model Manager & Tuning WebUI") as demo:
    gr.Markdown("# 🤖 LLM Model Manager & Tuning WebUI")
    
    if not LLM_AVAILABLE:
        gr.Markdown("⚠️ **LLM Engineering module not available.** Please install required dependencies.")
        gr.Markdown("Required: `llm_engineering` module with proper configuration")
    
    with gr.Tab("List Models"):
        gr.Markdown("## List All Models")
        list_btn = gr.Button("List Models")
        list_out = gr.Textbox(label="Loaded Models", lines=10)
        list_btn.click(list_models, outputs=list_out)
    
    with gr.Tab("Add Model"):
        gr.Markdown("## Add New Model")
        name = gr.Textbox(label="Model Name", placeholder="e.g., my_llama_model")
        provider_type = gr.Dropdown(
            choices=["llama_factory", "openai", "huggingface"], 
            label="Provider Type", 
            value="llama_factory"
        )
        config_json = gr.Textbox(
            label="Provider Config (JSON)", 
            lines=4,
            placeholder='{"model_name": "Llama-3-8B", "model_path": null}'
        )
        add_btn = gr.Button("Add Model")
        add_out = gr.Textbox(label="Result", lines=5)
        add_btn.click(add_model, inputs=[name, provider_type, config_json], outputs=add_out)
    
    with gr.Tab("Remove Model"):
        gr.Markdown("## Remove Model")
        rm_name = gr.Textbox(label="Model Name", placeholder="Enter model name to remove")
        rm_btn = gr.Button("Remove Model", variant="stop")
        rm_out = gr.Textbox(label="Result", lines=3)
        rm_btn.click(remove_model, inputs=rm_name, outputs=rm_out)
    
    with gr.Tab("Update Model Config"):
        gr.Markdown("## Update Model Configuration")
        up_name = gr.Textbox(label="Model Name", placeholder="Enter model name to update")
        up_config_json = gr.Textbox(
            label="New Config (JSON)", 
            lines=4,
            placeholder='{"model_name": "Llama-3-8B-v2", "model_path": "/path/to/model"}'
        )
        up_btn = gr.Button("Update Model")
        up_out = gr.Textbox(label="Result", lines=3)
        up_btn.click(update_model, inputs=[up_name, up_config_json], outputs=up_out)
    
    with gr.Tab("Fine-tune Model"):
        gr.Markdown("## Fine-tune Model")
        ft_name = gr.Textbox(label="Model Name", placeholder="Enter model name to fine-tune")
        ft_dataset = gr.Textbox(label="Dataset Path", placeholder="path/to/dataset.csv")
        ft_kwargs = gr.Textbox(
            label="Extra kwargs (JSON, optional)", 
            lines=2,
            placeholder='{"epochs": 3, "batch_size": 4}'
        )
        ft_btn = gr.Button("Fine-tune")
        ft_out = gr.Textbox(label="Result/Log", lines=6)
        ft_btn.click(fine_tune_model, inputs=[ft_name, ft_dataset, ft_kwargs], outputs=ft_out)
    
    with gr.Tab("Evaluate Model"):
        gr.Markdown("## Evaluate Model")
        ev_name = gr.Textbox(label="Model Name", placeholder="Enter model name to evaluate")
        ev_dataset = gr.Textbox(label="Dataset Path", placeholder="path/to/eval_dataset.csv")
        ev_kwargs = gr.Textbox(
            label="Extra kwargs (JSON, optional)", 
            lines=2,
            placeholder='{"metrics": ["accuracy", "f1"]}'
        )
        ev_btn = gr.Button("Evaluate")
        ev_out = gr.Textbox(label="Result/Log", lines=6)
        ev_btn.click(evaluate_model, inputs=[ev_name, ev_dataset, ev_kwargs], outputs=ev_out)

    with gr.Tab("Workflows (Ecommerce Support)"):
        gr.Markdown("## Run Ecommerce Customer Support Workflows (LangGraph)")
        with gr.Tab("Order Status"):
            customer_id = gr.Textbox(label="Customer ID", value="12345")
            order_id = gr.Textbox(label="Order ID", value="A1001")
            status = gr.Textbox(label="Order Status", value="Shipped")
            expected_delivery = gr.Textbox(label="Expected Delivery", value="2024-07-20")
            run_btn = gr.Button("Run Order Status Workflow")
            result_out = gr.Textbox(label="Result", lines=6)
            run_btn.click(run_order_status_workflow_ui, inputs=[customer_id, order_id, status, expected_delivery], outputs=result_out)
        with gr.Tab("Returns/Refunds"):
            customer_id2 = gr.Textbox(label="Customer ID", value="12345")
            order_id2 = gr.Textbox(label="Order ID", value="A1001")
            reason = gr.Textbox(label="Return Reason", value="Changed my mind")
            run_btn2 = gr.Button("Run Returns Workflow")
            result_out2 = gr.Textbox(label="Result", lines=6)
            run_btn2.click(run_returns_workflow_ui, inputs=[customer_id2, order_id2, reason], outputs=result_out2)
        with gr.Tab("Escalation"):
            customer_id3 = gr.Textbox(label="Customer ID", value="12345")
            issue = gr.Textbox(label="Escalation Issue", value="Order not delivered after 2 weeks")
            run_btn3 = gr.Button("Run Escalation Workflow")
            result_out3 = gr.Textbox(label="Result", lines=6)
            run_btn3.click(run_escalation_workflow_ui, inputs=[customer_id3, issue], outputs=result_out3)

    with gr.Tab("Workflow Logs"):
        gr.Markdown("## Recent Workflow Runs")
        log_btn = gr.Button("Refresh Logs")
        log_out = gr.Textbox(label="Workflow Logs", lines=20)
        def format_logs():
            logs = get_workflow_logs()
            if not logs:
                return "No workflow runs yet."
            out = []
            for entry in logs[-50:]:
                out.append(f"[{entry['timestamp']}] {entry['workflow']}\nInput: {entry['input']}\nOutput: {entry['output']}\nError: {entry['error']}\n---")
            return "\n".join(out)
        log_btn.click(lambda: format_logs(), outputs=log_out)
        # Show logs on load
        log_out.value = format_logs()

def main():
    parser = argparse.ArgumentParser(description="LLM Model Manager & Tuning WebUI")
    parser.add_argument("--server_port", type=int, default=7862, help="Server port")
    parser.add_argument("--server_name", type=str, default="0.0.0.0", help="Server name")
    args = parser.parse_args()
    
    print(f"Starting LLM WebUI on http://{args.server_name}:{args.server_port}")
    demo.launch(server_name=args.server_name, server_port=args.server_port)

if __name__ == "__main__":
    main() 