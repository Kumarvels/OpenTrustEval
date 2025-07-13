#!/usr/bin/env python3
"""
Hugging Face Integration WebUI for OpenTrustEval
Gradio-based web interface for ColBERT-v2 and other Hugging Face models
"""

import os
import sys
import json
import gradio as gr
import time
from typing import Dict, Any, List

# Add parent directories to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from llm_engineering.providers.huggingface_provider import HuggingFaceProvider
    from llm_engineering.llm_lifecycle import LLMLifecycleManager
    LLM_AVAILABLE = True
except ImportError as e:
    LLM_AVAILABLE = False
    print(f"Warning: LLM engineering not available: {e}")

# Initialize components
manager = None
if LLM_AVAILABLE:
    try:
        manager = LLMLifecycleManager()
    except Exception as e:
        print(f"Warning: Failed to initialize LLM manager: {e}")

# Available models for the WebUI
AVAILABLE_MODELS = {
    'colbert_v2_retrieval': {
        'name': 'ColBERT-v2 Retrieval',
        'type': 'retrieval',
        'description': 'State-of-the-art document retrieval model'
    },
    'gpt2_generation': {
        'name': 'GPT-2 Text Generation',
        'type': 'generation',
        'description': 'Text generation model'
    },
    't5_translation': {
        'name': 'T5 Translation',
        'type': 'seq2seq',
        'description': 'Text-to-text transfer model'
    },
    'bert_classification': {
        'name': 'BERT Classification',
        'type': 'classification',
        'description': 'Text classification model'
    },
    'dialogpt_chat': {
        'name': 'DialoGPT Chat',
        'type': 'generation',
        'description': 'Conversational AI model'
    },
    'bart_summarization': {
        'name': 'BART Summarization',
        'type': 'seq2seq',
        'description': 'Text summarization model'
    }
}

def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get information about a specific model"""
    if not manager or model_name not in manager.llm_providers:
        return {'error': f'Model {model_name} not found'}
    
    try:
        provider = manager.llm_providers[model_name]
        return provider.get_model_info()
    except Exception as e:
        return {'error': str(e)}

def perform_retrieval(model_name: str, query: str, documents: str, top_k: int) -> str:
    """Perform document retrieval using ColBERT-v2"""
    if not manager or model_name not in manager.llm_providers:
        return "❌ Model not found"
    
    try:
        provider = manager.llm_providers[model_name]
        
        # Parse documents
        doc_list = [doc.strip() for doc in documents.split('\n') if doc.strip()]
        
        if not doc_list:
            return "❌ No documents provided"
        
        # Perform retrieval
        start_time = time.time()
        results = provider._retrieve(query, doc_list, top_k=top_k)
        end_time = time.time()
        
        # Format results
        output = f"🔍 Query: {query}\n"
        output += f"⏱️  Retrieval time: {end_time - start_time:.3f}s\n"
        output += f"📊 Found {len(results['results'])} results:\n\n"
        
        for i, result in enumerate(results['results']):
            output += f"{i+1}. Score: {result['score']:.4f}\n"
            output += f"   Document: {result['document']}\n\n"
        
        return output
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

def generate_text(model_name: str, prompt: str, max_length: int, temperature: float) -> str:
    """Generate text using a language model"""
    if not manager or model_name not in manager.llm_providers:
        return "❌ Model not found"
    
    try:
        provider = manager.llm_providers[model_name]
        
        # Generate text
        start_time = time.time()
        generated_text = provider.generate(
            prompt, 
            max_length=max_length, 
            temperature=temperature
        )
        end_time = time.time()
        
        output = f"📝 Prompt: {prompt}\n"
        output += f"⏱️  Generation time: {end_time - start_time:.3f}s\n"
        output += f"🤖 Generated: {generated_text}\n"
        
        return output
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

def translate_text(model_name: str, text: str, task: str) -> str:
    """Translate or transform text using seq2seq models"""
    if not manager or model_name not in manager.llm_providers:
        return "❌ Model not found"
    
    try:
        provider = manager.llm_providers[model_name]
        
        # Prepare input based on task
        if task == "translate_en_fr":
            input_text = f"translate English to French: {text}"
        elif task == "translate_fr_en":
            input_text = f"translate French to English: {text}"
        elif task == "summarize":
            input_text = f"summarize: {text}"
        else:
            input_text = text
        
        # Generate
        start_time = time.time()
        result = provider.generate(input_text, max_length=100, temperature=0.7)
        end_time = time.time()
        
        output = f"📝 Input: {text}\n"
        output += f"🔄 Task: {task}\n"
        output += f"⏱️  Processing time: {end_time - start_time:.3f}s\n"
        output += f"📤 Result: {result}\n"
        
        return output
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

def classify_text(model_name: str, text: str) -> str:
    """Classify text using BERT or similar models"""
    if not manager or model_name not in manager.llm_providers:
        return "❌ Model not found"
    
    try:
        provider = manager.llm_providers[model_name]
        
        # For classification, we need to use the model directly
        if not provider.model or not provider.tokenizer:
            return "❌ Model not loaded properly"
        
        import torch
        
        # Prepare inputs
        inputs = provider.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(provider.device)
        
        # Get predictions
        start_time = time.time()
        with torch.no_grad():
            outputs = provider.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
        end_time = time.time()
        
        # Get top predictions
        top_probs, top_indices = torch.topk(predictions[0], 3)
        
        output = f"📝 Text: {text}\n"
        output += f"⏱️  Classification time: {end_time - start_time:.3f}s\n"
        output += f"🏷️  Top predictions:\n"
        
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            output += f"  {i+1}. Class {idx.item()}: {prob.item():.4f}\n"
        
        return output
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

def list_available_models() -> str:
    """List all available models"""
    if not manager:
        return "❌ LLM Manager not available"
    
    try:
        models = manager.list_models()
        output = "📋 Available Models:\n\n"
        
        for model_name in models:
            if model_name in AVAILABLE_MODELS:
                info = AVAILABLE_MODELS[model_name]
                output += f"🔹 {info['name']} ({model_name})\n"
                output += f"   Type: {info['type']}\n"
                output += f"   Description: {info['description']}\n\n"
            else:
                output += f"🔹 {model_name}\n\n"
        
        return output
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

def get_system_status() -> str:
    """Get system status and component availability"""
    status = "🔧 System Status:\n\n"
    
    # Check LLM availability
    status += f"📊 LLM Engineering: {'✅ Available' if LLM_AVAILABLE else '❌ Not Available'}\n"
    
    # Check manager
    status += f"🔗 LLM Manager: {'✅ Initialized' if manager else '❌ Not Initialized'}\n"
    
    # Check available models
    if manager:
        models = manager.list_models()
        status += f"🤖 Loaded Models: {len(models)}\n"
        for model in models:
            status += f"  - {model}\n"
    
    # Check high-performance components
    try:
        from llm_engineering.providers.huggingface_provider import get_high_performance_hf_status
        hf_status = get_high_performance_hf_status()
        status += f"\n🚀 High-Performance System: {'✅ Available' if hf_status['available'] else '❌ Not Available'}\n"
    except:
        status += f"\n🚀 High-Performance System: ❌ Not Available\n"
    
    return status

# Create Gradio interface
with gr.Blocks(title="Hugging Face Integration WebUI - OpenTrustEval") as demo:
    gr.Markdown("# 🤗 Hugging Face Integration WebUI")
    gr.Markdown("## OpenTrustEval - ColBERT-v2 and Model Integration")
    
    if not LLM_AVAILABLE:
        gr.Markdown("⚠️ **LLM Engineering module not available.** Please install required dependencies.")
        gr.Markdown("Required: `llm_engineering` module with proper configuration")
    
    with gr.Tab("🔍 Document Retrieval (ColBERT-v2)"):
        gr.Markdown("## ColBERT-v2 Document Retrieval")
        gr.Markdown("Use ColBERT-v2 for efficient document retrieval with late interaction.")
        
        with gr.Row():
            with gr.Column():
                retrieval_model = gr.Dropdown(
                    choices=["colbert_v2_retrieval"],
                    value="colbert_v2_retrieval",
                    label="Retrieval Model"
                )
                query = gr.Textbox(
                    label="Search Query",
                    placeholder="Enter your search query...",
                    lines=2
                )
                documents = gr.Textbox(
                    label="Documents (one per line)",
                    placeholder="Enter documents to search in, one per line...",
                    lines=10
                )
                top_k = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Top K Results"
                )
                retrieval_btn = gr.Button("🔍 Perform Retrieval", variant="primary")
            
            with gr.Column():
                retrieval_output = gr.Textbox(
                    label="Retrieval Results",
                    lines=15
                )
        
        retrieval_btn.click(
            perform_retrieval,
            inputs=[retrieval_model, query, documents, top_k],
            outputs=retrieval_output
        )
    
    with gr.Tab("🤖 Text Generation"):
        gr.Markdown("## Text Generation Models")
        gr.Markdown("Generate text using various language models.")
        
        with gr.Row():
            with gr.Column():
                gen_model = gr.Dropdown(
                    choices=["gpt2_generation", "dialogpt_chat"],
                    value="gpt2_generation",
                    label="Generation Model"
                )
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt...",
                    lines=3
                )
                max_length = gr.Slider(
                    minimum=10,
                    maximum=200,
                    value=50,
                    step=10,
                    label="Max Length"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature"
                )
                gen_btn = gr.Button("🤖 Generate Text", variant="primary")
            
            with gr.Column():
                gen_output = gr.Textbox(
                    label="Generated Text",
                    lines=10
                )
        
        gen_btn.click(
            generate_text,
            inputs=[gen_model, prompt, max_length, temperature],
            outputs=gen_output
        )
    
    with gr.Tab("🔄 Translation & Summarization"):
        gr.Markdown("## Seq2Seq Models")
        gr.Markdown("Translate text or generate summaries using T5 and BART.")
        
        with gr.Row():
            with gr.Column():
                seq2seq_model = gr.Dropdown(
                    choices=["t5_translation", "bart_summarization"],
                    value="t5_translation",
                    label="Seq2Seq Model"
                )
                task = gr.Dropdown(
                    choices=["translate_en_fr", "translate_fr_en", "summarize"],
                    value="translate_en_fr",
                    label="Task"
                )
                input_text = gr.Textbox(
                    label="Input Text",
                    placeholder="Enter text to translate or summarize...",
                    lines=4
                )
                seq2seq_btn = gr.Button("🔄 Process Text", variant="primary")
            
            with gr.Column():
                seq2seq_output = gr.Textbox(
                    label="Result",
                    lines=10
                )
        
        seq2seq_btn.click(
            translate_text,
            inputs=[seq2seq_model, input_text, task],
            outputs=seq2seq_output
        )
    
    with gr.Tab("🏷️ Text Classification"):
        gr.Markdown("## Text Classification")
        gr.Markdown("Classify text using BERT and similar models.")
        
        with gr.Row():
            with gr.Column():
                class_model = gr.Dropdown(
                    choices=["bert_classification"],
                    value="bert_classification",
                    label="Classification Model"
                )
                class_text = gr.Textbox(
                    label="Text to Classify",
                    placeholder="Enter text to classify...",
                    lines=3
                )
                class_btn = gr.Button("🏷️ Classify Text", variant="primary")
            
            with gr.Column():
                class_output = gr.Textbox(
                    label="Classification Results",
                    lines=10
                )
        
        class_btn.click(
            classify_text,
            inputs=[class_model, class_text],
            outputs=class_output
        )
    
    with gr.Tab("📋 Model Information"):
        gr.Markdown("## Model Information and System Status")
        
        with gr.Row():
            with gr.Column():
                model_name = gr.Dropdown(
                    choices=list(AVAILABLE_MODELS.keys()),
                    value="colbert_v2_retrieval",
                    label="Select Model"
                )
                info_btn = gr.Button("📊 Get Model Info", variant="primary")
                list_btn = gr.Button("📋 List All Models", variant="secondary")
                status_btn = gr.Button("🔧 System Status", variant="secondary")
            
            with gr.Column():
                info_output = gr.Textbox(
                    label="Information",
                    lines=15
                )
        
        info_btn.click(
            lambda x: json.dumps(get_model_info(x), indent=2),
            inputs=model_name,
            outputs=info_output
        )
        
        list_btn.click(
            list_available_models,
            outputs=info_output
        )
        
        status_btn.click(
            get_system_status,
            outputs=info_output
        )

def main():
    """Launch the WebUI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hugging Face Integration WebUI")
    parser.add_argument("--server_port", type=int, default=7863, help="Server port")
    parser.add_argument("--server_name", type=str, default="0.0.0.0", help="Server name")
    args = parser.parse_args()
    
    print(f"🚀 Starting Hugging Face WebUI on http://{args.server_name}:{args.server_port}")
    demo.launch(server_name=args.server_name, server_port=args.server_port)

if __name__ == "__main__":
    main() 