import os
import sys
import gradio as gr
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from llm_engineering.llm_lifecycle import LLMLifecycleManager

manager = LLMLifecycleManager()

# --- Model Management Functions ---
def list_models():
    return "\n".join(manager.list_models())

def add_model(name, provider_type, config_json):
    try:
        config = json.loads(config_json)
        manager.add_model(name, provider_type, config, persist=True)
        return f"Added model '{name}' of type '{provider_type}'."
    except Exception as e:
        return f"Error: {e}"

def remove_model(name):
    try:
        manager.remove_model(name, persist=True)
        return f"Removed model '{name}'."
    except Exception as e:
        return f"Error: {e}"

def update_model(name, config_json):
    try:
        config = json.loads(config_json)
        manager.update_model(name, config, persist=True)
        return f"Updated model '{name}'."
    except Exception as e:
        return f"Error: {e}"

# --- Tuning Functions ---
def fine_tune_model(name, dataset_path, extra_kwargs_json):
    try:
        provider = manager.llm_providers.get(name)
        if provider is None:
            return f"Provider '{name}' not found."
        extra_kwargs = json.loads(extra_kwargs_json) if extra_kwargs_json else {}
        result = provider.fine_tune(dataset_path, **extra_kwargs)
        return f"Fine-tuning result: {result}"
    except Exception as e:
        return f"Error: {e}"

def evaluate_model(name, dataset_path, extra_kwargs_json):
    try:
        provider = manager.llm_providers.get(name)
        if provider is None:
            return f"Provider '{name}' not found."
        extra_kwargs = json.loads(extra_kwargs_json) if extra_kwargs_json else {}
        result = provider.evaluate(dataset_path, **extra_kwargs)
        return f"Evaluation result: {result}"
    except Exception as e:
        return f"Error: {e}"

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# LLM Model Manager & Tuning WebUI")
    with gr.Tab("List Models"):
        list_btn = gr.Button("List Models")
        list_out = gr.Textbox(label="Loaded Models", lines=10)
        list_btn.click(list_models, outputs=list_out)
    with gr.Tab("Add Model"):
        name = gr.Textbox(label="Model Name")
        provider_type = gr.Textbox(label="Provider Type (e.g., llama_factory)")
        config_json = gr.Textbox(label="Provider Config (JSON)", lines=4)
        add_btn = gr.Button("Add Model")
        add_out = gr.Textbox(label="Result")
        add_btn.click(add_model, inputs=[name, provider_type, config_json], outputs=add_out)
    with gr.Tab("Remove Model"):
        rm_name = gr.Textbox(label="Model Name")
        rm_btn = gr.Button("Remove Model")
        rm_out = gr.Textbox(label="Result")
        rm_btn.click(remove_model, inputs=rm_name, outputs=rm_out)
    with gr.Tab("Update Model Config"):
        up_name = gr.Textbox(label="Model Name")
        up_config_json = gr.Textbox(label="New Config (JSON)", lines=4)
        up_btn = gr.Button("Update Model")
        up_out = gr.Textbox(label="Result")
        up_btn.click(update_model, inputs=[up_name, up_config_json], outputs=up_out)
    with gr.Tab("Fine-tune Model"):
        ft_name = gr.Textbox(label="Model Name")
        ft_dataset = gr.Textbox(label="Dataset Path")
        ft_kwargs = gr.Textbox(label="Extra kwargs (JSON, optional)", lines=2)
        ft_btn = gr.Button("Fine-tune")
        ft_out = gr.Textbox(label="Result/Log", lines=4)
        ft_btn.click(fine_tune_model, inputs=[ft_name, ft_dataset, ft_kwargs], outputs=ft_out)
    with gr.Tab("Evaluate Model"):
        ev_name = gr.Textbox(label="Model Name")
        ev_dataset = gr.Textbox(label="Dataset Path")
        ev_kwargs = gr.Textbox(label="Extra kwargs (JSON, optional)", lines=2)
        ev_btn = gr.Button("Evaluate")
        ev_out = gr.Textbox(label="Result/Log", lines=4)
        ev_btn.click(evaluate_model, inputs=[ev_name, ev_dataset, ev_kwargs], outputs=ev_out)

demo.launch() 