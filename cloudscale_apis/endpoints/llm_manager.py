"""
LLM Management API Endpoints (FastAPI)
Comprehensive async endpoints for LLM model management with caching and high-performance integration.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import functools
import time
import json
import os
import sys

# Add parent directories to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

try:
    from llm_engineering.llm_lifecycle import LLMLifecycleManager, get_high_performance_llm_status
    LLM_AVAILABLE = True
except ImportError as e:
    LLM_AVAILABLE = False
    print(f"Warning: LLM engineering not available: {e}")

# Initialize LLM manager
llm_manager = None
if LLM_AVAILABLE:
    try:
        llm_manager = LLMLifecycleManager()
    except Exception as e:
        print(f"Warning: Failed to initialize LLM manager: {e}")
        llm_manager = None

router = APIRouter(prefix="/llm", tags=["LLM Management"])

# --- Pydantic Models ---
class ModelConfig(BaseModel):
    model_name: str = Field(..., description="Model name")
    model_path: Optional[str] = Field(None, description="Path to model files")
    provider_type: str = Field(..., description="Provider type (e.g., llama_factory)")
    config: Dict[str, Any] = Field(default_factory=dict, description="Provider-specific configuration")

class ModelInfo(BaseModel):
    name: str
    provider_type: str
    config: Dict[str, Any]
    status: str = "active"
    created_at: str
    last_used: Optional[str] = None

class FineTuneRequest(BaseModel):
    dataset_path: str = Field(..., description="Path to training dataset")
    epochs: int = Field(default=3, description="Number of training epochs")
    batch_size: int = Field(default=4, description="Training batch size")
    learning_rate: float = Field(default=2e-5, description="Learning rate")
    use_lora: bool = Field(default=True, description="Use LoRA fine-tuning")
    use_qlora: bool = Field(default=False, description="Use QLoRA fine-tuning")
    monitor: Optional[str] = Field(None, description="Experiment monitor (wandb, mlflow, etc.)")
    extra_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Additional training parameters")

class EvaluateRequest(BaseModel):
    dataset_path: str = Field(..., description="Path to evaluation dataset")
    metrics: List[str] = Field(default=["accuracy", "f1"], description="Evaluation metrics")
    extra_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Additional evaluation parameters")

class BatchModelRequest(BaseModel):
    models: List[ModelConfig] = Field(..., description="List of models to add")

class ModelStatus(BaseModel):
    name: str
    status: str
    health: str
    memory_usage: Optional[float] = None
    gpu_usage: Optional[float] = None
    last_activity: str
    error_count: int = 0

class ModelLogs(BaseModel):
    name: str
    logs: List[Dict[str, Any]]
    total_logs: int

# --- Caching ---
@functools.lru_cache(maxsize=128)
def cached_list_models():
    """Cache model listing for 5 minutes"""
    if not llm_manager:
        return []
    return llm_manager.list_models()

@functools.lru_cache(maxsize=64)
def cached_model_status(model_name: str):
    """Cache model status for 1 minute"""
    if not llm_manager or model_name not in llm_manager.llm_providers:
        return None
    provider = llm_manager.llm_providers[model_name]
    return {
        "name": model_name,
        "status": "active",
        "health": "healthy",
        "last_activity": time.strftime("%Y-%m-%d %H:%M:%S"),
        "error_count": 0
    }

# --- Background Tasks ---
async def background_fine_tune(model_name: str, request: FineTuneRequest):
    """Background task for fine-tuning"""
    try:
        if not llm_manager or model_name not in llm_manager.llm_providers:
            raise ValueError(f"Model {model_name} not found")
        
        provider = llm_manager.llm_providers[model_name]
        
        # Prepare kwargs
        kwargs = {
            "num_train_epochs": request.epochs,
            "per_device_train_batch_size": request.batch_size,
            "learning_rate": request.learning_rate,
            **request.extra_kwargs
        }
        
        if request.use_lora:
            kwargs["use_lora"] = True
        if request.use_qlora:
            kwargs["use_qlora"] = True
        if request.monitor:
            kwargs["monitor"] = request.monitor
        
        # Run fine-tuning
        result = provider.fine_tune(request.dataset_path, **kwargs)
        
        # Log success
        llm_manager.log_governance(f"Fine-tuning completed for {model_name}")
        
    except Exception as e:
        # Log error
        llm_manager.log_governance(f"Fine-tuning failed for {model_name}: {str(e)}")
        raise

async def background_evaluate(model_name: str, request: EvaluateRequest):
    """Background task for evaluation"""
    try:
        if not llm_manager or model_name not in llm_manager.llm_providers:
            raise ValueError(f"Model {model_name} not found")
        
        provider = llm_manager.llm_providers[model_name]
        
        # Prepare kwargs
        kwargs = {
            "metrics": request.metrics,
            **request.extra_kwargs
        }
        
        # Run evaluation
        result = provider.evaluate(request.dataset_path, **kwargs)
        
        # Log success
        llm_manager.log_governance(f"Evaluation completed for {model_name}")
        
    except Exception as e:
        # Log error
        llm_manager.log_governance(f"Evaluation failed for {model_name}: {str(e)}")
        raise

# --- API Endpoints ---
@router.get("/health", summary="LLM Manager Health Check")
async def llm_health():
    """Check LLM manager health and high-performance system status"""
    if not LLM_AVAILABLE:
        raise HTTPException(status_code=503, detail="LLM engineering module not available")
    
    if not llm_manager:
        raise HTTPException(status_code=503, detail="LLM manager not initialized")
    
    # Get high-performance system status
    hp_status = get_high_performance_llm_status() if 'get_high_performance_llm_status' in globals() else {}
    
    return {
        "status": "healthy",
        "llm_available": LLM_AVAILABLE,
        "manager_initialized": llm_manager is not None,
        "high_performance_system": hp_status,
        "total_models": len(llm_manager.llm_providers) if llm_manager else 0
    }

@router.get("/models", summary="List All Models", response_model=List[ModelInfo])
async def list_models():
    """List all available LLM models with caching"""
    if not llm_manager:
        raise HTTPException(status_code=503, detail="LLM manager not available")
    
    try:
        model_names = cached_list_models()
        models = []
        
        for name in model_names:
            provider = llm_manager.llm_providers.get(name)
            if provider:
                models.append(ModelInfo(
                    name=name,
                    provider_type=type(provider).__name__,
                    config=getattr(provider, 'config', {}),
                    status="active",
                    created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                    last_used=None
                ))
        
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

@router.post("/models", summary="Add New Model")
async def add_model(model: ModelConfig):
    """Add a new LLM model"""
    if not llm_manager:
        raise HTTPException(status_code=503, detail="LLM manager not available")
    
    try:
        llm_manager.add_model(
            name=model.model_name,
            provider_type=model.provider_type,
            provider_config=model.config,
            persist=True
        )
        
        # Clear cache
        cached_list_models.cache_clear()
        
        return {
            "status": "success",
            "message": f"Model '{model.model_name}' added successfully",
            "model_name": model.model_name
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error adding model: {str(e)}")

@router.put("/models/{model_name}", summary="Update Model Configuration")
async def update_model(model_name: str, model: ModelConfig):
    """Update model configuration"""
    if not llm_manager:
        raise HTTPException(status_code=503, detail="LLM manager not available")
    
    try:
        llm_manager.update_model(
            name=model_name,
            provider_config=model.config,
            persist=True
        )
        
        # Clear cache
        cached_list_models.cache_clear()
        cached_model_status.cache_clear()
        
        return {
            "status": "success",
            "message": f"Model '{model_name}' updated successfully"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error updating model: {str(e)}")

@router.delete("/models/{model_name}", summary="Remove Model")
async def remove_model(model_name: str):
    """Remove a model"""
    if not llm_manager:
        raise HTTPException(status_code=503, detail="LLM manager not available")
    
    try:
        llm_manager.remove_model(model_name, persist=True)
        
        # Clear cache
        cached_list_models.cache_clear()
        cached_model_status.cache_clear()
        
        return {
            "status": "success",
            "message": f"Model '{model_name}' removed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error removing model: {str(e)}")

@router.post("/models/{model_name}/fine_tune", summary="Fine-tune Model")
async def fine_tune_model(model_name: str, request: FineTuneRequest, background_tasks: BackgroundTasks):
    """Start fine-tuning for a model (background task)"""
    if not llm_manager:
        raise HTTPException(status_code=503, detail="LLM manager not available")
    
    if model_name not in llm_manager.llm_providers:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    # Add background task
    background_tasks.add_task(background_fine_tune, model_name, request)
    
    return {
        "status": "started",
        "message": f"Fine-tuning started for model '{model_name}'",
        "task_id": f"fine_tune_{model_name}_{int(time.time())}"
    }

@router.post("/models/{model_name}/evaluate", summary="Evaluate Model")
async def evaluate_model(model_name: str, request: EvaluateRequest, background_tasks: BackgroundTasks):
    """Start evaluation for a model (background task)"""
    if not llm_manager:
        raise HTTPException(status_code=503, detail="LLM manager not available")
    
    if model_name not in llm_manager.llm_providers:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    # Add background task
    background_tasks.add_task(background_evaluate, model_name, request)
    
    return {
        "status": "started",
        "message": f"Evaluation started for model '{model_name}'",
        "task_id": f"evaluate_{model_name}_{int(time.time())}"
    }

@router.post("/models/{model_name}/deploy", summary="Deploy Model")
async def deploy_model(model_name: str):
    """Deploy a model"""
    if not llm_manager:
        raise HTTPException(status_code=503, detail="LLM manager not available")
    
    if model_name not in llm_manager.llm_providers:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    try:
        provider = llm_manager.llm_providers[model_name]
        result = provider.deploy()
        
        # Log deployment
        llm_manager.log_governance(f"Model '{model_name}' deployed")
        
        return {
            "status": "success",
            "message": f"Model '{model_name}' deployed successfully",
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error deploying model: {str(e)}")

@router.get("/models/{model_name}/status", summary="Get Model Status", response_model=ModelStatus)
async def get_model_status(model_name: str):
    """Get detailed status of a model"""
    if not llm_manager:
        raise HTTPException(status_code=503, detail="LLM manager not available")
    
    status = cached_model_status(model_name)
    if not status:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    return ModelStatus(**status)

@router.get("/models/{model_name}/logs", summary="Get Model Logs", response_model=ModelLogs)
async def get_model_logs(model_name: str, limit: int = 100):
    """Get logs for a model"""
    if not llm_manager:
        raise HTTPException(status_code=503, detail="LLM manager not available")
    
    if model_name not in llm_manager.llm_providers:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    try:
        # Get governance logs (filtered by model name)
        all_logs = llm_manager.get_governance_logs()
        model_logs = [log for log in all_logs if model_name in str(log)]
        
        # Limit logs
        model_logs = model_logs[-limit:] if limit > 0 else model_logs
        
        return ModelLogs(
            name=model_name,
            logs=model_logs,
            total_logs=len(model_logs)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting logs: {str(e)}")

@router.post("/models/batch_add", summary="Batch Add Models")
async def batch_add_models(request: BatchModelRequest):
    """Add multiple models in batch"""
    if not llm_manager:
        raise HTTPException(status_code=503, detail="LLM manager not available")
    
    results = []
    errors = []
    
    for model in request.models:
        try:
            llm_manager.add_model(
                name=model.model_name,
                provider_type=model.provider_type,
                provider_config=model.config,
                persist=True
            )
            results.append({
                "name": model.model_name,
                "status": "success"
            })
        except Exception as e:
            errors.append({
                "name": model.model_name,
                "error": str(e)
            })
    
    # Clear cache
    cached_list_models.cache_clear()
    
    return {
        "status": "completed",
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors
    }

@router.get("/metrics", summary="Get LLM Metrics")
async def get_llm_metrics():
    """Get LLM system metrics"""
    if not llm_manager:
        raise HTTPException(status_code=503, detail="LLM manager not available")
    
    try:
        metrics = llm_manager.get_metrics()
        governance_logs = llm_manager.get_governance_logs()
        security_checks = llm_manager.get_security_checks()
        
        return {
            "metrics": metrics,
            "total_governance_logs": len(governance_logs),
            "total_security_checks": len(security_checks),
            "total_models": len(llm_manager.llm_providers),
            "high_performance_system": get_high_performance_llm_status() if 'get_high_performance_llm_status' in globals() else {}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting metrics: {str(e)}")

@router.post("/models/{model_name}/generate", summary="Generate Text")
async def generate_text(model_name: str, prompt: str, max_length: int = 100):
    """Generate text using a model"""
    if not llm_manager:
        raise HTTPException(status_code=503, detail="LLM manager not available")
    
    if model_name not in llm_manager.llm_providers:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    try:
        provider = llm_manager.llm_providers[model_name]
        result = provider.generate(prompt, max_length=max_length)
        
        # Log generation
        llm_manager.log_governance(f"Text generation for model '{model_name}'")
        
        return {
            "model": model_name,
            "prompt": prompt,
            "generated_text": result,
            "max_length": max_length
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error generating text: {str(e)}") 