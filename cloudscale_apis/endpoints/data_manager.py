from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional, Union
import asyncio
import pandas as pd
import json
import logging
from datetime import datetime
import aiofiles
import aiofiles.os
from pathlib import Path
import hashlib
import pickle
from functools import lru_cache
import time

# Import data engineering components
try:
    # from data_engineering.data_lifecycle import DataLifecycleManager
    # from data_engineering.dataset_integration import DatasetManager, DatasetConnector
    # from data_engineering.advanced_trust_scoring import AdvancedTrustScoringEngine
    DATA_ENGINEERING_AVAILABLE = False
    print("Warning: Data engineering components temporarily disabled")
except ImportError as e:
    print(f"Warning: Data engineering components not available: {e}")
    DATA_ENGINEERING_AVAILABLE = False

router = APIRouter(prefix="/data", tags=["Data Management"])

# Global instances with caching
_data_manager = None
_dataset_manager = None
_trust_scorer = None
_cache = {}
_cache_timestamps = {}

def get_data_manager():
    """Get or create DataLifecycleManager instance"""
    global _data_manager
    if _data_manager is None and DATA_ENGINEERING_AVAILABLE:
        _data_manager = DataLifecycleManager()
    return _data_manager

def get_dataset_manager():
    """Get or create DatasetManager instance"""
    global _dataset_manager
    if _dataset_manager is None and DATA_ENGINEERING_AVAILABLE:
        _dataset_manager = DatasetManager()
    return _dataset_manager

def get_trust_scorer():
    """Get or create AdvancedTrustScoringEngine instance"""
    global _trust_scorer
    if _trust_scorer is None and DATA_ENGINEERING_AVAILABLE:
        _trust_scorer = AdvancedTrustScoringEngine()
    return _trust_scorer

# Cache management
def get_cache_key(operation: str, params: Dict) -> str:
    """Generate cache key for operation"""
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(f"{operation}:{param_str}".encode()).hexdigest()

def is_cache_valid(cache_key: str, max_age: int = 300) -> bool:
    """Check if cache entry is still valid"""
    if cache_key not in _cache_timestamps:
        return False
    return time.time() - _cache_timestamps[cache_key] < max_age

def set_cache(cache_key: str, data: Any):
    """Set cache entry with timestamp"""
    _cache[cache_key] = data
    _cache_timestamps[cache_key] = time.time()

def get_cache(cache_key: str):
    """Get cache entry if valid"""
    if is_cache_valid(cache_key):
        return _cache.get(cache_key)
    return None

# Health check
@router.get("/health")
async def data_health_check():
    """Health check for data management system"""
    try:
        data_manager = get_data_manager()
        dataset_manager = get_dataset_manager()
        
        return {
            "status": "healthy",
            "data_engineering_available": DATA_ENGINEERING_AVAILABLE,
            "data_manager_ready": data_manager is not None,
            "dataset_manager_ready": dataset_manager is not None,
            "cache_entries": len(_cache),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Dataset CRUD Operations
@router.post("/datasets/create")
async def create_dataset(
    name: str = Form(...),
    data: Optional[str] = Form(None),
    schema: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """Create a new dataset"""
    try:
        dataset_manager = get_dataset_manager()
        if not dataset_manager:
            raise HTTPException(status_code=503, detail="Dataset manager not available")
        
        # Parse optional parameters
        schema_dict = json.loads(schema) if schema else None
        metadata_dict = json.loads(metadata) if metadata else None
        
        # Handle file upload
        if file:
            # Save uploaded file temporarily
            temp_path = f"/tmp/{file.filename}"
            async with aiofiles.open(temp_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # Import dataset from file
            dataset_id = dataset_manager.import_dataset(temp_path, name)
            
            # Clean up temp file
            await aiofiles.os.remove(temp_path)
        else:
            # Create from provided data
            if not data:
                raise HTTPException(status_code=400, detail="Either file or data must be provided")
            
            data_dict = json.loads(data)
            dataset_id = dataset_manager.create_dataset(name, data_dict, schema_dict, metadata_dict)
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "name": name,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create dataset: {str(e)}")

@router.get("/datasets/list")
async def list_datasets(
    format: str = Query("json", description="Output format: json or csv"),
    limit: int = Query(100, description="Maximum number of datasets to return"),
    offset: int = Query(0, description="Number of datasets to skip")
):
    """List all datasets with pagination"""
    try:
        dataset_manager = get_dataset_manager()
        if not dataset_manager:
            raise HTTPException(status_code=503, detail="Dataset manager not available")
        
        # Check cache
        cache_key = get_cache_key("list_datasets", {"format": format, "limit": limit, "offset": offset})
        cached_result = get_cache(cache_key)
        if cached_result:
            return cached_result
        
        datasets = dataset_manager.list_datasets()
        
        # Apply pagination
        total_count = len(datasets)
        datasets = datasets[offset:offset + limit]
        
        result = {
            "success": True,
            "datasets": datasets,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "timestamp": datetime.now().isoformat()
        }
        
        # Set cache
        set_cache(cache_key, result)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {str(e)}")

@router.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get dataset details and metadata"""
    try:
        dataset_manager = get_dataset_manager()
        if not dataset_manager:
            raise HTTPException(status_code=503, detail="Dataset manager not available")
        
        # Check cache
        cache_key = get_cache_key("get_dataset", {"dataset_id": dataset_id})
        cached_result = get_cache(cache_key)
        if cached_result:
            return cached_result
        
        # Get dataset info
        dataset_info = dataset_manager.datasets.get(dataset_id)
        if not dataset_info:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        # Load metadata
        try:
            with open(dataset_info['metadata_file'], 'r') as f:
                metadata = json.load(f)
        except:
            metadata = {}
        
        # Load schema
        try:
            with open(dataset_info['schema_file'], 'r') as f:
                schema = json.load(f)
        except:
            schema = {}
        
        result = {
            "success": True,
            "dataset_id": dataset_id,
            "info": dataset_info,
            "metadata": metadata,
            "schema": schema,
            "timestamp": datetime.now().isoformat()
        }
        
        # Set cache
        set_cache(cache_key, result)
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dataset: {str(e)}")

@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str, force: bool = Query(False, description="Force deletion")):
    """Delete a dataset"""
    try:
        dataset_manager = get_dataset_manager()
        if not dataset_manager:
            raise HTTPException(status_code=503, detail="Dataset manager not available")
        
        success = dataset_manager.delete_dataset(dataset_id)
        
        if not success and not force:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
        
        # Clear related cache entries
        cache_keys_to_clear = [k for k in _cache.keys() if dataset_id in k]
        for key in cache_keys_to_clear:
            _cache.pop(key, None)
            _cache_timestamps.pop(key, None)
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "deleted": success,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")

# Dataset Operations
@router.post("/datasets/{dataset_id}/load")
async def load_dataset(dataset_id: str, preview: bool = Query(True, description="Return preview only")):
    """Load dataset data"""
    try:
        dataset_manager = get_dataset_manager()
        if not dataset_manager:
            raise HTTPException(status_code=503, detail="Dataset manager not available")
        
        # Check cache
        cache_key = get_cache_key("load_dataset", {"dataset_id": dataset_id, "preview": preview})
        cached_result = get_cache(cache_key)
        if cached_result:
            return cached_result
        
        df = dataset_manager.load_dataset(dataset_id)
        
        if preview:
            data = df.head(100).to_dict('records')
            total_rows = len(df)
        else:
            data = df.to_dict('records')
            total_rows = len(df)
        
        result = {
            "success": True,
            "dataset_id": dataset_id,
            "data": data,
            "total_rows": total_rows,
            "columns": list(df.columns),
            "preview": preview,
            "timestamp": datetime.now().isoformat()
        }
        
        # Set cache
        set_cache(cache_key, result)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {str(e)}")

@router.post("/datasets/{dataset_id}/validate")
async def validate_dataset(
    dataset_id: str,
    validation_rules: Optional[str] = Form(None, description="JSON validation rules")
):
    """Validate a dataset"""
    try:
        dataset_manager = get_dataset_manager()
        if not dataset_manager:
            raise HTTPException(status_code=503, detail="Dataset manager not available")
        
        # Parse validation rules
        rules = json.loads(validation_rules) if validation_rules else None
        
        # Check cache
        cache_key = get_cache_key("validate_dataset", {"dataset_id": dataset_id, "rules": rules})
        cached_result = get_cache(cache_key)
        if cached_result:
            return cached_result
        
        results = dataset_manager.validate_dataset(dataset_id, rules)
        
        result = {
            "success": True,
            "dataset_id": dataset_id,
            "validation_results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Set cache
        set_cache(cache_key, result)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to validate dataset: {str(e)}")

@router.post("/datasets/{dataset_id}/process")
async def process_dataset(
    dataset_id: str,
    transformations: str = Form(..., description="JSON transformations array")
):
    """Process dataset with transformations"""
    try:
        dataset_manager = get_dataset_manager()
        if not dataset_manager:
            raise HTTPException(status_code=503, detail="Dataset manager not available")
        
        # Parse transformations
        transforms = json.loads(transformations)
        
        new_dataset_id = dataset_manager.process_dataset(dataset_id, transforms)
        
        return {
            "success": True,
            "original_dataset_id": dataset_id,
            "new_dataset_id": new_dataset_id,
            "transformations": transforms,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process dataset: {str(e)}")

@router.post("/datasets/{dataset_id}/export")
async def export_dataset(
    dataset_id: str,
    format: str = Form(..., description="Export format: csv, json, parquet, excel"),
    output_path: Optional[str] = Form(None, description="Output file path")
):
    """Export dataset to various formats"""
    try:
        dataset_manager = get_dataset_manager()
        if not dataset_manager:
            raise HTTPException(status_code=503, detail="Dataset manager not available")
        
        output_file = dataset_manager.export_dataset(dataset_id, format, output_path)
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "format": format,
            "output_file": output_file,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export dataset: {str(e)}")

# Trust Scoring Operations
@router.post("/datasets/{dataset_id}/trust-score")
async def calculate_trust_score(
    dataset_id: str,
    method: str = Form("ensemble", description="Trust scoring method")
):
    """Calculate trust score for dataset"""
    try:
        dataset_manager = get_dataset_manager()
        trust_scorer = get_trust_scorer()
        
        if not dataset_manager:
            raise HTTPException(status_code=503, detail="Dataset manager not available")
        if not trust_scorer:
            raise HTTPException(status_code=503, detail="Trust scorer not available")
        
        # Check cache
        cache_key = get_cache_key("trust_score", {"dataset_id": dataset_id, "method": method})
        cached_result = get_cache(cache_key)
        if cached_result:
            return cached_result
        
        # Load dataset
        df = dataset_manager.load_dataset(dataset_id)
        
        # Calculate trust score
        if method == "ensemble":
            trust_score = trust_scorer.calculate_ensemble_trust_score(df)
        elif method == "robust":
            trust_score = trust_scorer.calculate_robust_trust_score(df)
        elif method == "uncertainty":
            trust_score = trust_scorer.calculate_uncertainty_trust_score(df)
        else:
            trust_score = trust_scorer.calculate_advanced_trust_score(df)
        
        result = {
            "success": True,
            "dataset_id": dataset_id,
            "method": method,
            "trust_score": trust_score,
            "timestamp": datetime.now().isoformat()
        }
        
        # Set cache
        set_cache(cache_key, result)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate trust score: {str(e)}")

@router.post("/datasets/{dataset_id}/quality-filter")
async def create_quality_filtered_dataset(
    dataset_id: str,
    min_trust_score: float = Form(0.8, description="Minimum trust score threshold"),
    features: Optional[str] = Form(None, description="JSON array of features to include")
):
    """Create quality-filtered dataset"""
    try:
        dataset_manager = get_dataset_manager()
        if not dataset_manager:
            raise HTTPException(status_code=503, detail="Dataset manager not available")
        
        # Parse features
        feature_list = json.loads(features) if features else None
        
        new_dataset_id = dataset_manager.create_quality_filtered_dataset(
            dataset_id, min_trust_score, feature_list
        )
        
        return {
            "success": True,
            "original_dataset_id": dataset_id,
            "new_dataset_id": new_dataset_id,
            "min_trust_score": min_trust_score,
            "features": feature_list,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create quality filtered dataset: {str(e)}")

# Batch Operations
@router.post("/batch/trust-score")
async def batch_trust_scoring(
    dataset_ids: str = Form(..., description="JSON array of dataset IDs"),
    method: str = Form("ensemble", description="Trust scoring method")
):
    """Batch trust scoring for multiple datasets"""
    try:
        dataset_manager = get_dataset_manager()
        trust_scorer = get_trust_scorer()
        
        if not dataset_manager:
            raise HTTPException(status_code=503, detail="Dataset manager not available")
        if not trust_scorer:
            raise HTTPException(status_code=503, detail="Trust scorer not available")
        
        # Parse dataset IDs
        ids = json.loads(dataset_ids)
        
        # Check cache
        cache_key = get_cache_key("batch_trust_score", {"dataset_ids": ids, "method": method})
        cached_result = get_cache(cache_key)
        if cached_result:
            return cached_result
        
        results = []
        
        # Process datasets in parallel
        async def process_single_dataset(dataset_id):
            try:
                df = dataset_manager.load_dataset(dataset_id)
                if method == "ensemble":
                    trust_score = trust_scorer.calculate_ensemble_trust_score(df)
                elif method == "robust":
                    trust_score = trust_scorer.calculate_robust_trust_score(df)
                elif method == "uncertainty":
                    trust_score = trust_scorer.calculate_uncertainty_trust_score(df)
                else:
                    trust_score = trust_scorer.calculate_advanced_trust_score(df)
                
                return {
                    "dataset_id": dataset_id,
                    "trust_score": trust_score,
                    "status": "success"
                }
            except Exception as e:
                return {
                    "dataset_id": dataset_id,
                    "error": str(e),
                    "status": "error"
                }
        
        # Execute batch processing
        tasks = [process_single_dataset(dataset_id) for dataset_id in ids]
        results = await asyncio.gather(*tasks)
        
        result = {
            "success": True,
            "method": method,
            "results": results,
            "total_datasets": len(ids),
            "successful": len([r for r in results if r["status"] == "success"]),
            "failed": len([r for r in results if r["status"] == "error"]),
            "timestamp": datetime.now().isoformat()
        }
        
        # Set cache
        set_cache(cache_key, result)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to perform batch trust scoring: {str(e)}")

@router.post("/batch/validate")
async def batch_validation(
    dataset_ids: str = Form(..., description="JSON array of dataset IDs"),
    validation_rules: Optional[str] = Form(None, description="JSON validation rules")
):
    """Batch validation for multiple datasets"""
    try:
        dataset_manager = get_dataset_manager()
        if not dataset_manager:
            raise HTTPException(status_code=503, detail="Dataset manager not available")
        
        # Parse parameters
        ids = json.loads(dataset_ids)
        rules = json.loads(validation_rules) if validation_rules else None
        
        # Check cache
        cache_key = get_cache_key("batch_validate", {"dataset_ids": ids, "rules": rules})
        cached_result = get_cache(cache_key)
        if cached_result:
            return cached_result
        
        results = []
        
        # Process datasets in parallel
        async def validate_single_dataset(dataset_id):
            try:
                validation_result = dataset_manager.validate_dataset(dataset_id, rules)
                return {
                    "dataset_id": dataset_id,
                    "validation_results": validation_result,
                    "status": "success"
                }
            except Exception as e:
                return {
                    "dataset_id": dataset_id,
                    "error": str(e),
                    "status": "error"
                }
        
        # Execute batch validation
        tasks = [validate_single_dataset(dataset_id) for dataset_id in ids]
        results = await asyncio.gather(*tasks)
        
        result = {
            "success": True,
            "results": results,
            "total_datasets": len(ids),
            "successful": len([r for r in results if r["status"] == "success"]),
            "failed": len([r for r in results if r["status"] == "error"]),
            "timestamp": datetime.now().isoformat()
        }
        
        # Set cache
        set_cache(cache_key, result)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to perform batch validation: {str(e)}")

# Data Engineering Operations
@router.post("/pipeline/run")
async def run_data_pipeline(
    pipeline_config: str = Form(..., description="JSON pipeline configuration")
):
    """Run data engineering pipeline"""
    try:
        data_manager = get_data_manager()
        if not data_manager:
            raise HTTPException(status_code=503, detail="Data manager not available")
        
        # Parse pipeline configuration
        config = json.loads(pipeline_config)
        
        # Run pipeline
        result = data_manager.run_pipeline(config.get('steps', []))
        
        return {
            "success": True,
            "pipeline_result": result,
            "metrics": data_manager.get_metrics(),
            "governance_logs": data_manager.get_governance_logs(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run pipeline: {str(e)}")

@router.get("/pipeline/status")
async def get_pipeline_status():
    """Get data pipeline status and metrics"""
    try:
        data_manager = get_data_manager()
        if not data_manager:
            raise HTTPException(status_code=503, detail="Data manager not available")
        
        return {
            "success": True,
            "metrics": data_manager.get_metrics(),
            "governance_logs": data_manager.get_governance_logs(),
            "connectors": list(data_manager.connectors.keys()),
            "databases": list(data_manager.databases.keys()),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline status: {str(e)}")

# Cache Management
@router.post("/cache/clear")
async def clear_cache():
    """Clear all cache entries"""
    try:
        global _cache, _cache_timestamps
        cache_count = len(_cache)
        _cache.clear()
        _cache_timestamps.clear()
        
        return {
            "success": True,
            "cleared_entries": cache_count,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@router.get("/cache/status")
async def get_cache_status():
    """Get cache status and statistics"""
    try:
        return {
            "success": True,
            "cache_entries": len(_cache),
            "cache_size_mb": sum(len(pickle.dumps(v)) for v in _cache.values()) / (1024 * 1024),
            "oldest_entry": min(_cache_timestamps.values()) if _cache_timestamps else None,
            "newest_entry": max(_cache_timestamps.values()) if _cache_timestamps else None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache status: {str(e)}")

# System Information
@router.get("/system/info")
async def get_system_info():
    """Get data management system information"""
    try:
        return {
            "success": True,
            "data_engineering_available": DATA_ENGINEERING_AVAILABLE,
            "data_manager_ready": get_data_manager() is not None,
            "dataset_manager_ready": get_dataset_manager() is not None,
            "trust_scorer_ready": get_trust_scorer() is not None,
            "cache_entries": len(_cache),
            "supported_formats": ["csv", "json", "parquet", "excel", "pickle", "feather"],
            "supported_operations": [
                "create", "read", "update", "delete",
                "validate", "process", "export", "import",
                "trust_score", "quality_filter", "batch_operations"
            ],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system info: {str(e)}") 