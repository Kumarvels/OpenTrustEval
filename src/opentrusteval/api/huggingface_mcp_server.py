#!/usr/bin/env python3
"""
Hugging Face MCP Server for OpenTrustEval
Model Context Protocol server for Hugging Face model integration
"""

import asyncio
import json
import logging
import sys
import os
from typing import Any, Dict, List, Optional, Sequence
from dataclasses import dataclass
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from mcp import Server, StdioServerParameters
    from mcp.types import (
        CallToolRequest, CallToolResult, ListToolsRequest, ListToolsResult,
        Tool, TextContent, ImageContent, EmbeddedResource
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️ MCP library not available. Install with: pip install mcp")

try:
    from llm_engineering.providers.huggingface_provider import HuggingFaceProvider
    from llm_engineering.llm_lifecycle import LLMLifecycleManager
    LLM_AVAILABLE = True
except ImportError as e:
    LLM_AVAILABLE = False
    print(f"⚠️ LLM engineering not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Model information structure"""
    name: str
    type: str
    description: str
    config: Dict[str, Any]
    loaded: bool = False

class HuggingFaceMCPServer:
    """
    MCP Server for Hugging Face model integration with OpenTrustEval
    Provides standardized interface for model management and inference
    """
    
    def __init__(self):
        self.server = Server("huggingface-mcp-server")
        self.llm_manager = None
        self.models: Dict[str, ModelInfo] = {}
        self.active_models: Dict[str, HuggingFaceProvider] = {}
        
        # Initialize LLM manager if available
        if LLM_AVAILABLE:
            try:
                self.llm_manager = LLMLifecycleManager()
                self._load_configured_models()
            except Exception as e:
                logger.error(f"Failed to initialize LLM manager: {e}")
        
        # Register MCP tools
        self._register_tools()
    
    def _load_configured_models(self):
        """Load models from LLM manager configuration"""
        if not self.llm_manager:
            return
        
        try:
            # Get Hugging Face providers from LLM manager
            for provider_name, provider in self.llm_manager.llm_providers.items():
                if hasattr(provider, 'model_name') and hasattr(provider, 'model_type'):
                    model_info = ModelInfo(
                        name=provider_name,
                        type=provider.model_type,
                        description=f"Hugging Face {provider.model_type} model: {provider.model_name}",
                        config=provider.config,
                        loaded=hasattr(provider, 'model') and provider.model is not None
                    )
                    self.models[provider_name] = model_info
                    logger.info(f"Loaded model: {provider_name}")
        except Exception as e:
            logger.error(f"Error loading configured models: {e}")
    
    def _register_tools(self):
        """Register MCP tools"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> ListToolsResult:
            """List available tools"""
            tools = [
                Tool(
                    name="list_models",
                    description="List all available Hugging Face models",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="load_model",
                    description="Load a Hugging Face model",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "model_name": {
                                "type": "string",
                                "description": "Name of the model to load"
                            },
                            "model_type": {
                                "type": "string",
                                "enum": ["causal", "seq2seq", "classification", "retrieval"],
                                "description": "Type of model"
                            },
                            "config": {
                                "type": "object",
                                "description": "Model configuration"
                            }
                        },
                        "required": ["model_name", "model_type"]
                    }
                ),
                Tool(
                    name="generate_text",
                    description="Generate text using a loaded model",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "model_name": {
                                "type": "string",
                                "description": "Name of the loaded model"
                            },
                            "prompt": {
                                "type": "string",
                                "description": "Input prompt for text generation"
                            },
                            "max_length": {
                                "type": "integer",
                                "description": "Maximum length of generated text"
                            },
                            "temperature": {
                                "type": "number",
                                "description": "Temperature for text generation"
                            }
                        },
                        "required": ["model_name", "prompt"]
                    }
                ),
                Tool(
                    name="retrieve_documents",
                    description="Retrieve documents using ColBERT-v2 or other retrieval models",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "model_name": {
                                "type": "string",
                                "description": "Name of the retrieval model"
                            },
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "documents": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of documents to search in"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of top results to return"
                            }
                        },
                        "required": ["model_name", "query", "documents"]
                    }
                ),
                Tool(
                    name="classify_text",
                    description="Classify text using a classification model",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "model_name": {
                                "type": "string",
                                "description": "Name of the classification model"
                            },
                            "text": {
                                "type": "string",
                                "description": "Text to classify"
                            }
                        },
                        "required": ["model_name", "text"]
                    }
                ),
                Tool(
                    name="translate_text",
                    description="Translate text using a seq2seq model",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "model_name": {
                                "type": "string",
                                "description": "Name of the translation model"
                            },
                            "text": {
                                "type": "string",
                                "description": "Text to translate"
                            },
                            "task": {
                                "type": "string",
                                "enum": ["translate_en_fr", "translate_fr_en", "summarize"],
                                "description": "Translation task"
                            }
                        },
                        "required": ["model_name", "text", "task"]
                    }
                ),
                Tool(
                    name="get_model_info",
                    description="Get information about a loaded model",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "model_name": {
                                "type": "string",
                                "description": "Name of the model"
                            }
                        },
                        "required": ["model_name"]
                    }
                ),
                Tool(
                    name="fine_tune_model",
                    description="Fine-tune a model on a dataset",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "model_name": {
                                "type": "string",
                                "description": "Name of the model to fine-tune"
                            },
                            "dataset_path": {
                                "type": "string",
                                "description": "Path to the training dataset"
                            },
                            "epochs": {
                                "type": "integer",
                                "description": "Number of training epochs"
                            },
                            "batch_size": {
                                "type": "integer",
                                "description": "Training batch size"
                            }
                        },
                        "required": ["model_name", "dataset_path"]
                    }
                ),
                Tool(
                    name="evaluate_model",
                    description="Evaluate a model on test data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "model_name": {
                                "type": "string",
                                "description": "Name of the model to evaluate"
                            },
                            "eval_data": {
                                "type": "string",
                                "description": "Path to evaluation data"
                            }
                        },
                        "required": ["model_name", "eval_data"]
                    }
                )
            ]
            return ListToolsResult(tools=tools)
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """Handle tool calls"""
            try:
                if name == "list_models":
                    return await self._list_models()
                elif name == "load_model":
                    return await self._load_model(arguments)
                elif name == "generate_text":
                    return await self._generate_text(arguments)
                elif name == "retrieve_documents":
                    return await self._retrieve_documents(arguments)
                elif name == "classify_text":
                    return await self._classify_text(arguments)
                elif name == "translate_text":
                    return await self._translate_text(arguments)
                elif name == "get_model_info":
                    return await self._get_model_info(arguments)
                elif name == "fine_tune_model":
                    return await self._fine_tune_model(arguments)
                elif name == "evaluate_model":
                    return await self._evaluate_model(arguments)
                else:
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"Unknown tool: {name}")]
                    )
            except Exception as e:
                logger.error(f"Error in tool call {name}: {e}")
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {str(e)}")]
                )
    
    async def _list_models(self) -> CallToolResult:
        """List all available models"""
        if not self.models:
            return CallToolResult(
                content=[TextContent(type="text", text="No models available")]
            )
        
        model_list = []
        for name, info in self.models.items():
            status = "✅ Loaded" if info.loaded else "⏳ Not Loaded"
            model_list.append(f"- {name} ({info.type}): {status}")
        
        return CallToolResult(
            content=[TextContent(type="text", text="Available Models:\n" + "\n".join(model_list))]
        )
    
    async def _load_model(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Load a Hugging Face model"""
        model_name = arguments.get("model_name")
        model_type = arguments.get("model_type")
        config = arguments.get("config", {})
        
        if not model_name or not model_type:
            return CallToolResult(
                content=[TextContent(type="text", text="Error: model_name and model_type are required")]
            )
        
        try:
            # Create provider configuration
            provider_config = {
                'model_name': model_name,
                'model_type': model_type,
                'device': 'auto',
                **config
            }
            
            # Load the model
            provider = HuggingFaceProvider(provider_config)
            self.active_models[model_name] = provider
            
            # Update model info
            if model_name in self.models:
                self.models[model_name].loaded = True
            else:
                self.models[model_name] = ModelInfo(
                    name=model_name,
                    type=model_type,
                    description=f"Loaded {model_type} model: {model_name}",
                    config=provider_config,
                    loaded=True
                )
            
            return CallToolResult(
                content=[TextContent(type="text", text=f"✅ Successfully loaded model: {model_name}")]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"❌ Failed to load model {model_name}: {str(e)}")]
            )
    
    async def _generate_text(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Generate text using a loaded model"""
        model_name = arguments.get("model_name")
        prompt = arguments.get("prompt")
        max_length = arguments.get("max_length", 100)
        temperature = arguments.get("temperature", 0.7)
        
        if not model_name or not prompt:
            return CallToolResult(
                content=[TextContent(type="text", text="Error: model_name and prompt are required")]
            )
        
        if model_name not in self.active_models:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: Model {model_name} is not loaded")]
            )
        
        try:
            provider = self.active_models[model_name]
            generated_text = provider.generate(
                prompt, 
                max_length=max_length, 
                temperature=temperature
            )
            
            return CallToolResult(
                content=[TextContent(type="text", text=f"Generated Text:\n{generated_text}")]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"❌ Error generating text: {str(e)}")]
            )
    
    async def _retrieve_documents(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Retrieve documents using a retrieval model"""
        model_name = arguments.get("model_name")
        query = arguments.get("query")
        documents = arguments.get("documents", [])
        top_k = arguments.get("top_k", 5)
        
        if not model_name or not query or not documents:
            return CallToolResult(
                content=[TextContent(type="text", text="Error: model_name, query, and documents are required")]
            )
        
        if model_name not in self.active_models:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: Model {model_name} is not loaded")]
            )
        
        try:
            provider = self.active_models[model_name]
            results = provider._retrieve(query, documents, top_k=top_k)
            
            # Format results
            output = f"Query: {query}\n\nTop {len(results['results'])} Results:\n"
            for i, result in enumerate(results['results']):
                output += f"{i+1}. Score: {result['score']:.4f}\n   {result['document']}\n\n"
            
            return CallToolResult(
                content=[TextContent(type="text", text=output)]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"❌ Error retrieving documents: {str(e)}")]
            )
    
    async def _classify_text(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Classify text using a classification model"""
        model_name = arguments.get("model_name")
        text = arguments.get("text")
        
        if not model_name or not text:
            return CallToolResult(
                content=[TextContent(type="text", text="Error: model_name and text are required")]
            )
        
        if model_name not in self.active_models:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: Model {model_name} is not loaded")]
            )
        
        try:
            provider = self.active_models[model_name]
            
            # Use the model directly for classification
            inputs = provider.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(provider.device)
            
            import torch
            with torch.no_grad():
                outputs = provider.model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=-1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(predictions[0], 3)
            
            output = f"Text: {text}\n\nTop Predictions:\n"
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                output += f"{i+1}. Class {idx.item()}: {prob.item():.4f}\n"
            
            return CallToolResult(
                content=[TextContent(type="text", text=output)]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"❌ Error classifying text: {str(e)}")]
            )
    
    async def _translate_text(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Translate text using a seq2seq model"""
        model_name = arguments.get("model_name")
        text = arguments.get("text")
        task = arguments.get("task", "translate_en_fr")
        
        if not model_name or not text:
            return CallToolResult(
                content=[TextContent(type="text", text="Error: model_name and text are required")]
            )
        
        if model_name not in self.active_models:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: Model {model_name} is not loaded")]
            )
        
        try:
            provider = self.active_models[model_name]
            
            # Prepare input based on task
            if task == "translate_en_fr":
                input_text = f"translate English to French: {text}"
            elif task == "translate_fr_en":
                input_text = f"translate French to English: {text}"
            elif task == "summarize":
                input_text = f"summarize: {text}"
            else:
                input_text = text
            
            result = provider.generate(input_text, max_length=100, temperature=0.7)
            
            return CallToolResult(
                content=[TextContent(type="text", text=f"Input: {text}\nTask: {task}\nResult: {result}")]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"❌ Error translating text: {str(e)}")]
            )
    
    async def _get_model_info(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Get information about a model"""
        model_name = arguments.get("model_name")
        
        if not model_name:
            return CallToolResult(
                content=[TextContent(type="text", text="Error: model_name is required")]
            )
        
        if model_name in self.active_models:
            provider = self.active_models[model_name]
            info = provider.get_model_info()
            
            output = f"Model Information for {model_name}:\n"
            for key, value in info.items():
                output += f"- {key}: {value}\n"
            
            return CallToolResult(
                content=[TextContent(type="text", text=output)]
            )
        else:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Model {model_name} is not loaded")]
            )
    
    async def _fine_tune_model(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Fine-tune a model"""
        model_name = arguments.get("model_name")
        dataset_path = arguments.get("dataset_path")
        epochs = arguments.get("epochs", 3)
        batch_size = arguments.get("batch_size", 8)
        
        if not model_name or not dataset_path:
            return CallToolResult(
                content=[TextContent(type="text", text="Error: model_name and dataset_path are required")]
            )
        
        if model_name not in self.active_models:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: Model {model_name} is not loaded")]
            )
        
        try:
            provider = self.active_models[model_name]
            result = provider.fine_tune(
                dataset_path,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size
            )
            
            return CallToolResult(
                content=[TextContent(type="text", text=f"Fine-tuning completed: {result}")]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"❌ Error fine-tuning model: {str(e)}")]
            )
    
    async def _evaluate_model(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Evaluate a model"""
        model_name = arguments.get("model_name")
        eval_data = arguments.get("eval_data")
        
        if not model_name or not eval_data:
            return CallToolResult(
                content=[TextContent(type="text", text="Error: model_name and eval_data are required")]
            )
        
        if model_name not in self.active_models:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: Model {model_name} is not loaded")]
            )
        
        try:
            provider = self.active_models[model_name]
            results = provider.evaluate(eval_data)
            
            output = f"Evaluation Results for {model_name}:\n"
            for key, value in results.items():
                output += f"- {key}: {value}\n"
            
            return CallToolResult(
                content=[TextContent(type="text", text=output)]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"❌ Error evaluating model: {str(e)}")]
            )
    
    async def run(self):
        """Run the MCP server"""
        if not MCP_AVAILABLE:
            logger.error("MCP library not available")
            return
        
        logger.info("Starting Hugging Face MCP Server...")
        
        # Create server parameters
        params = StdioServerParameters()
        
        # Run the server
        async with self.server.run_stdio(params) as stream:
            logger.info("MCP Server started successfully")
            await stream.wait_closed()

async def main():
    """Main entry point"""
    server = HuggingFaceMCPServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main()) 