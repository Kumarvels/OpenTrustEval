from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional, Union
import asyncio
import json
import logging
from datetime import datetime
import hashlib
import pickle
from functools import lru_cache
import time
import uuid
from pathlib import Path
import pandas as pd
import numpy as np

# Import research components
try:
    from high_performance_system.core.ultimate_moe_system import UltimateMoESystem
    from high_performance_system.core.advanced_expert_ensemble import AdvancedExpertEnsemble
    from high_performance_system.core.intelligent_domain_router import IntelligentDomainRouter
    RESEARCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Research components not available: {e}")
    RESEARCH_AVAILABLE = False

router = APIRouter(prefix="/research", tags=["Research Platform"])

# Global instances with caching
_moe_system = None
_expert_ensemble = None
_domain_router = None
_cache = {}
_cache_timestamps = {}

# Research settings
RESEARCH_CONFIG = {
    "max_experiments": 100,
    "max_dataset_size_mb": 100,
    "experiment_timeout_seconds": 3600,
    "max_concurrent_experiments": 10,
    "cache_ttl_seconds": 1800
}

# Experiment tracking
experiments = {}
experiment_results = {}
active_experiments = set()

def get_moe_system():
    """Get or create UltimateMoESystem instance"""
    global _moe_system
    if _moe_system is None and RESEARCH_AVAILABLE:
        _moe_system = UltimateMoESystem()
    return _moe_system

def get_expert_ensemble():
    """Get or create AdvancedExpertEnsemble instance"""
    global _expert_ensemble
    if _expert_ensemble is None and RESEARCH_AVAILABLE:
        _expert_ensemble = AdvancedExpertEnsemble()
    return _expert_ensemble

def get_domain_router():
    """Get or create IntelligentDomainRouter instance"""
    global _domain_router
    if _domain_router is None and RESEARCH_AVAILABLE:
        _domain_router = IntelligentDomainRouter()
    return _domain_router

# Cache management
def get_cache_key(operation: str, params: Dict) -> str:
    """Generate cache key for operation"""
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(f"{operation}:{param_str}".encode()).hexdigest()

def is_cache_valid(cache_key: str, max_age: int = RESEARCH_CONFIG["cache_ttl_seconds"]) -> bool:
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
async def research_health_check():
    """Health check for research platform"""
    try:
        moe_system = get_moe_system()
        expert_ensemble = get_expert_ensemble()
        domain_router = get_domain_router()
        
        return {
            "status": "healthy",
            "research_available": RESEARCH_AVAILABLE,
            "moe_system_ready": moe_system is not None,
            "expert_ensemble_ready": expert_ensemble is not None,
            "domain_router_ready": domain_router is not None,
            "cache_entries": len(_cache),
            "active_experiments": len(active_experiments),
            "total_experiments": len(experiments),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Research Assistant Operations
@router.post("/assistant/query")
async def research_assistant_query(
    query: str = Form(...),
    context: Optional[str] = Form(None),
    domain: Optional[str] = Form("general"),
    max_results: int = Query(10, description="Maximum number of results")
):
    """Query the research assistant"""
    try:
        # Check cache
        cache_key = get_cache_key("research_query", {
            "query": query, "context": context, "domain": domain, "max_results": max_results
        })
        cached_result = get_cache(cache_key)
        if cached_result:
            return cached_result
        
        moe_system = get_moe_system()
        if not moe_system:
            raise HTTPException(status_code=503, detail="MoE system not available")
        
        # Process query with MoE system
        combined_text = f"Query: {query}\nContext: {context or 'No context provided'}"
        result = await moe_system.verify_text(combined_text, domain)
        
        # Format response for research assistant
        response = {
            "query": query,
            "context": context,
            "domain": domain,
            "verification_result": {
                "verified": result.verification_score > 0.5,
                "confidence": result.confidence,
                "hallucination_risk": getattr(result, 'hallucination_risk', 1.0 - result.confidence),
                "primary_domain": result.primary_domain
            },
            "expert_analysis": {
                expert: {
                    "confidence": expert_result.confidence,
                    "verification_score": expert_result.verification_score,
                    "reasoning": expert_result.reasoning,
                    "sources_used": expert_result.sources_used
                }
                for expert, expert_result in result.expert_results.items()
            },
            "sources": result.sources_used or [],
            "recommendations": _generate_research_recommendations(result, query),
            "timestamp": datetime.now().isoformat()
        }
        
        # Set cache
        set_cache(cache_key, response)
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process research query: {str(e)}")

def _generate_research_recommendations(result, query: str) -> List[str]:
    """Generate research recommendations based on verification result"""
    recommendations = []
    
    if result.confidence < 0.7:
        recommendations.append("Consider additional fact-checking for this query")
    
    if result.verification_score < 0.6:
        recommendations.append("Multiple sources should be consulted for verification")
    
    if len(result.sources_used or []) < 2:
        recommendations.append("Expand source diversity for more comprehensive analysis")
    
    if result.primary_domain != "general":
        recommendations.append(f"Consult domain-specific experts in {result.primary_domain}")
    
    return recommendations

# Use Case Creator Operations
@router.post("/usecase/create")
async def create_use_case(
    title: str = Form(...),
    description: str = Form(...),
    domain: str = Form(...),
    requirements: Optional[str] = Form(None),
    tags: Optional[str] = Form(None)
):
    """Create a new research use case"""
    try:
        use_case_id = str(uuid.uuid4())
        
        # Parse optional parameters
        requirements_dict = json.loads(requirements) if requirements else {}
        tags_list = json.loads(tags) if tags else []
        
        use_case = {
            "id": use_case_id,
            "title": title,
            "description": description,
            "domain": domain,
            "requirements": requirements_dict,
            "tags": tags_list,
            "status": "created",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "experiments": [],
            "results": []
        }
        
        experiments[use_case_id] = use_case
        
        return {
            "success": True,
            "use_case_id": use_case_id,
            "use_case": use_case,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create use case: {str(e)}")

@router.get("/usecase/list")
async def list_use_cases(
    domain: Optional[str] = Query(None, description="Filter by domain"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, description="Maximum number of use cases to return"),
    offset: int = Query(0, description="Number of use cases to skip")
):
    """List all use cases"""
    try:
        # Check cache
        cache_key = get_cache_key("list_use_cases", {
            "domain": domain, "status": status, "limit": limit, "offset": offset
        })
        cached_result = get_cache(cache_key)
        if cached_result:
            return cached_result
        
        # Filter use cases
        filtered_cases = list(experiments.values())
        
        if domain:
            filtered_cases = [uc for uc in filtered_cases if uc["domain"] == domain]
        
        if status:
            filtered_cases = [uc for uc in filtered_cases if uc["status"] == status]
        
        # Apply pagination
        total_count = len(filtered_cases)
        filtered_cases = filtered_cases[offset:offset + limit]
        
        result = {
            "success": True,
            "use_cases": filtered_cases,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "timestamp": datetime.now().isoformat()
        }
        
        # Set cache
        set_cache(cache_key, result)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list use cases: {str(e)}")

@router.get("/usecase/{use_case_id}")
async def get_use_case(use_case_id: str):
    """Get use case details"""
    try:
        # Check cache
        cache_key = get_cache_key("get_use_case", {"use_case_id": use_case_id})
        cached_result = get_cache(cache_key)
        if cached_result:
            return cached_result
        
        use_case = experiments.get(use_case_id)
        if not use_case:
            raise HTTPException(status_code=404, detail="Use case not found")
        
        result = {
            "success": True,
            "use_case": use_case,
            "timestamp": datetime.now().isoformat()
        }
        
        # Set cache
        set_cache(cache_key, result)
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get use case: {str(e)}")

@router.put("/usecase/{use_case_id}")
async def update_use_case(
    use_case_id: str,
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    status: Optional[str] = Form(None),
    requirements: Optional[str] = Form(None),
    tags: Optional[str] = Form(None)
):
    """Update use case"""
    try:
        use_case = experiments.get(use_case_id)
        if not use_case:
            raise HTTPException(status_code=404, detail="Use case not found")
        
        # Update fields
        if title:
            use_case["title"] = title
        if description:
            use_case["description"] = description
        if status:
            use_case["status"] = status
        if requirements:
            use_case["requirements"] = json.loads(requirements)
        if tags:
            use_case["tags"] = json.loads(tags)
        
        use_case["updated_at"] = datetime.now().isoformat()
        
        return {
            "success": True,
            "use_case": use_case,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update use case: {str(e)}")

@router.delete("/usecase/{use_case_id}")
async def delete_use_case(use_case_id: str):
    """Delete use case"""
    try:
        if use_case_id not in experiments:
            raise HTTPException(status_code=404, detail="Use case not found")
        
        # Remove from active experiments if running
        if use_case_id in active_experiments:
            active_experiments.remove(use_case_id)
        
        # Delete use case and results
        del experiments[use_case_id]
        experiment_results.pop(use_case_id, None)
        
        return {
            "success": True,
            "deleted": True,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete use case: {str(e)}")

# Experiment Lab Operations
@router.post("/experiment/create")
async def create_experiment(
    use_case_id: str = Form(...),
    name: str = Form(...),
    description: str = Form(...),
    parameters: Optional[str] = Form(None),
    dataset: Optional[UploadFile] = File(None)
):
    """Create a new experiment"""
    try:
        # Check if use case exists
        if use_case_id not in experiments:
            raise HTTPException(status_code=404, detail="Use case not found")
        
        # Check concurrent experiment limit
        if len(active_experiments) >= RESEARCH_CONFIG["max_concurrent_experiments"]:
            raise HTTPException(status_code=429, detail="Maximum concurrent experiments reached")
        
        experiment_id = str(uuid.uuid4())
        
        # Parse parameters
        parameters_dict = json.loads(parameters) if parameters else {}
        
        # Handle dataset upload
        dataset_info = None
        if dataset:
            # Check file size
            content = await dataset.read()
            if len(content) > RESEARCH_CONFIG["max_dataset_size_mb"] * 1024 * 1024:
                raise HTTPException(status_code=400, detail="Dataset file too large")
            
            dataset_info = {
                "filename": dataset.filename,
                "size_bytes": len(content),
                "content_type": dataset.content_type
            }
        
        experiment = {
            "id": experiment_id,
            "use_case_id": use_case_id,
            "name": name,
            "description": description,
            "parameters": parameters_dict,
            "dataset": dataset_info,
            "status": "created",
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "results": None,
            "metrics": {}
        }
        
        # Add to use case
        experiments[use_case_id]["experiments"].append(experiment_id)
        
        # Store experiment
        experiment_results[experiment_id] = experiment
        
        return {
            "success": True,
            "experiment_id": experiment_id,
            "experiment": experiment,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create experiment: {str(e)}")

@router.post("/experiment/{experiment_id}/run")
async def run_experiment(
    experiment_id: str,
    background_tasks: BackgroundTasks
):
    """Run an experiment"""
    try:
        experiment = experiment_results.get(experiment_id)
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        if experiment["status"] == "running":
            raise HTTPException(status_code=400, detail="Experiment already running")
        
        if experiment["status"] == "completed":
            raise HTTPException(status_code=400, detail="Experiment already completed")
        
        # Check concurrent experiment limit
        if len(active_experiments) >= RESEARCH_CONFIG["max_concurrent_experiments"]:
            raise HTTPException(status_code=429, detail="Maximum concurrent experiments reached")
        
        # Update experiment status
        experiment["status"] = "running"
        experiment["started_at"] = datetime.now().isoformat()
        active_experiments.add(experiment_id)
        
        # Start experiment in background
        background_tasks.add_task(_run_experiment_background, experiment_id)
        
        return {
            "success": True,
            "experiment_id": experiment_id,
            "status": "started",
            "message": "Experiment started successfully",
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run experiment: {str(e)}")

async def _run_experiment_background(experiment_id: str):
    """Run experiment in background"""
    try:
        experiment = experiment_results[experiment_id]
        moe_system = get_moe_system()
        
        if not moe_system:
            experiment["status"] = "failed"
            experiment["results"] = {"error": "MoE system not available"}
            return
        
        # Simulate experiment execution
        await asyncio.sleep(5)  # Simulate processing time
        
        # Run experiment based on parameters
        results = await _execute_experiment(experiment, moe_system)
        
        # Update experiment
        experiment["status"] = "completed"
        experiment["completed_at"] = datetime.now().isoformat()
        experiment["results"] = results
        
        # Remove from active experiments
        active_experiments.discard(experiment_id)
        
    except Exception as e:
        experiment["status"] = "failed"
        experiment["results"] = {"error": str(e)}
        active_experiments.discard(experiment_id)

async def _execute_experiment(experiment: Dict, moe_system) -> Dict:
    """Execute experiment based on parameters"""
    parameters = experiment.get("parameters", {})
    
    # Default test query
    test_query = parameters.get("test_query", "This is a test query for research purposes")
    domain = parameters.get("domain", "general")
    
    # Run verification
    result = await moe_system.verify_text(test_query, domain)
    
    # Calculate metrics
    metrics = {
        "verification_score": result.verification_score,
        "confidence": result.confidence,
        "hallucination_risk": getattr(result, 'hallucination_risk', 1.0 - result.confidence),
        "expert_count": len(result.expert_results),
        "sources_count": len(result.sources_used or []),
        "primary_domain": result.primary_domain
    }
    
    # Expert analysis
    expert_analysis = {}
    for expert_name, expert_result in result.expert_results.items():
        expert_analysis[expert_name] = {
            "confidence": expert_result.confidence,
            "verification_score": expert_result.verification_score,
            "reasoning": expert_result.reasoning,
            "sources_used": expert_result.sources_used
        }
    
    return {
        "metrics": metrics,
        "expert_analysis": expert_analysis,
        "sources_used": result.sources_used or [],
        "ensemble_verification": result.ensemble_verification,
        "routing_result": result.routing_result.__dict__ if result.routing_result else {},
        "test_query": test_query,
        "domain": domain
    }

@router.get("/experiment/{experiment_id}")
async def get_experiment(experiment_id: str):
    """Get experiment details and results"""
    try:
        experiment = experiment_results.get(experiment_id)
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        return {
            "success": True,
            "experiment": experiment,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get experiment: {str(e)}")

@router.get("/experiment/list")
async def list_experiments(
    use_case_id: Optional[str] = Query(None, description="Filter by use case"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, description="Maximum number of experiments to return"),
    offset: int = Query(0, description="Number of experiments to skip")
):
    """List all experiments"""
    try:
        # Check cache
        cache_key = get_cache_key("list_experiments", {
            "use_case_id": use_case_id, "status": status, "limit": limit, "offset": offset
        })
        cached_result = get_cache(cache_key)
        if cached_result:
            return cached_result
        
        # Filter experiments
        filtered_experiments = list(experiment_results.values())
        
        if use_case_id:
            filtered_experiments = [exp for exp in filtered_experiments if exp["use_case_id"] == use_case_id]
        
        if status:
            filtered_experiments = [exp for exp in filtered_experiments if exp["status"] == status]
        
        # Apply pagination
        total_count = len(filtered_experiments)
        filtered_experiments = filtered_experiments[offset:offset + limit]
        
        result = {
            "success": True,
            "experiments": filtered_experiments,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "active_experiments": list(active_experiments),
            "timestamp": datetime.now().isoformat()
        }
        
        # Set cache
        set_cache(cache_key, result)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list experiments: {str(e)}")

# Analysis Tools Operations
@router.post("/analysis/compare")
async def compare_experiments(
    experiment_ids: str = Form(..., description="JSON array of experiment IDs")
):
    """Compare multiple experiments"""
    try:
        # Parse experiment IDs
        ids = json.loads(experiment_ids)
        
        # Get experiments
        experiments_to_compare = []
        for exp_id in ids:
            experiment = experiment_results.get(exp_id)
            if not experiment:
                raise HTTPException(status_code=404, detail=f"Experiment {exp_id} not found")
            experiments_to_compare.append(experiment)
        
        # Perform comparison
        comparison = _compare_experiments(experiments_to_compare)
        
        return {
            "success": True,
            "experiment_ids": ids,
            "comparison": comparison,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compare experiments: {str(e)}")

def _compare_experiments(experiments: List[Dict]) -> Dict:
    """Compare experiments and generate analysis"""
    comparison = {
        "metrics_comparison": {},
        "expert_analysis_comparison": {},
        "performance_analysis": {},
        "recommendations": []
    }
    
    # Compare metrics
    metrics = ["verification_score", "confidence", "hallucination_risk"]
    for metric in metrics:
        values = []
        for exp in experiments:
            if exp["results"] and "metrics" in exp["results"]:
                values.append(exp["results"]["metrics"].get(metric, 0))
        
        if values:
            comparison["metrics_comparison"][metric] = {
                "values": values,
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values)
            }
    
    # Compare expert analysis
    all_experts = set()
    for exp in experiments:
        if exp["results"] and "expert_analysis" in exp["results"]:
            all_experts.update(exp["results"]["expert_analysis"].keys())
    
    for expert in all_experts:
        expert_data = []
        for exp in experiments:
            if (exp["results"] and "expert_analysis" in exp["results"] and 
                expert in exp["results"]["expert_analysis"]):
                expert_data.append(exp["results"]["expert_analysis"][expert])
        
        if expert_data:
            comparison["expert_analysis_comparison"][expert] = {
                "confidence_values": [d["confidence"] for d in expert_data],
                "verification_values": [d["verification_score"] for d in expert_data],
                "mean_confidence": np.mean([d["confidence"] for d in expert_data]),
                "mean_verification": np.mean([d["verification_score"] for d in expert_data])
            }
    
    # Performance analysis
    completion_times = []
    for exp in experiments:
        if exp["started_at"] and exp["completed_at"]:
            start = datetime.fromisoformat(exp["started_at"])
            end = datetime.fromisoformat(exp["completed_at"])
            completion_times.append((end - start).total_seconds())
    
    if completion_times:
        comparison["performance_analysis"] = {
            "completion_times": completion_times,
            "mean_completion_time": np.mean(completion_times),
            "fastest_experiment": np.min(completion_times),
            "slowest_experiment": np.max(completion_times)
        }
    
    # Generate recommendations
    if comparison["metrics_comparison"]:
        best_verification = max(comparison["metrics_comparison"]["verification_score"]["values"])
        best_confidence = max(comparison["metrics_comparison"]["confidence"]["values"])
        
        comparison["recommendations"].append(f"Best verification score: {best_verification:.3f}")
        comparison["recommendations"].append(f"Best confidence score: {best_confidence:.3f}")
    
    return comparison

@router.post("/analysis/trend")
async def analyze_trends(
    use_case_id: str = Form(...),
    metric: str = Form("verification_score", description="Metric to analyze"),
    time_period: str = Form("7d", description="Time period: 1d, 7d, 30d, 90d")
):
    """Analyze trends for a use case"""
    try:
        use_case = experiments.get(use_case_id)
        if not use_case:
            raise HTTPException(status_code=404, detail="Use case not found")
        
        # Get experiments for this use case
        use_case_experiments = [
            exp for exp in experiment_results.values() 
            if exp["use_case_id"] == use_case_id and exp["status"] == "completed"
        ]
        
        if not use_case_experiments:
            raise HTTPException(status_code=404, detail="No completed experiments found")
        
        # Analyze trends
        trends = _analyze_trends(use_case_experiments, metric, time_period)
        
        return {
            "success": True,
            "use_case_id": use_case_id,
            "metric": metric,
            "time_period": time_period,
            "trends": trends,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze trends: {str(e)}")

def _analyze_trends(experiments: List[Dict], metric: str, time_period: str) -> Dict:
    """Analyze trends in experiment results"""
    # Sort experiments by completion time
    sorted_experiments = sorted(
        experiments, 
        key=lambda x: x["completed_at"] if x["completed_at"] else "1970-01-01"
    )
    
    # Extract metric values over time
    time_series = []
    for exp in sorted_experiments:
        if exp["results"] and "metrics" in exp["results"]:
            value = exp["results"]["metrics"].get(metric, 0)
            time_series.append({
                "timestamp": exp["completed_at"],
                "value": value,
                "experiment_id": exp["id"]
            })
    
    if not time_series:
        return {"error": "No data available for trend analysis"}
    
    # Calculate trend statistics
    values = [point["value"] for point in time_series]
    
    # Simple linear trend (you could use more sophisticated methods)
    x = np.arange(len(values))
    if len(values) > 1:
        slope, intercept = np.polyfit(x, values, 1)
        trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
    else:
        slope, intercept = 0, values[0] if values else 0
        trend_direction = "stable"
    
    return {
        "time_series": time_series,
        "trend_direction": trend_direction,
        "slope": slope,
        "intercept": intercept,
        "mean_value": np.mean(values),
        "std_value": np.std(values),
        "min_value": np.min(values),
        "max_value": np.max(values),
        "data_points": len(values)
    }

# Project Management Operations
@router.post("/project/create")
async def create_research_project(
    name: str = Form(...),
    description: str = Form(...),
    objectives: Optional[str] = Form(None),
    team_members: Optional[str] = Form(None),
    timeline: Optional[str] = Form(None)
):
    """Create a new research project"""
    try:
        project_id = str(uuid.uuid4())
        
        # Parse optional parameters
        objectives_list = json.loads(objectives) if objectives else []
        team_members_list = json.loads(team_members) if team_members else []
        timeline_dict = json.loads(timeline) if timeline else {}
        
        project = {
            "id": project_id,
            "name": name,
            "description": description,
            "objectives": objectives_list,
            "team_members": team_members_list,
            "timeline": timeline_dict,
            "status": "planning",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "use_cases": [],
            "experiments": [],
            "progress": 0.0
        }
        
        # Store project (you might want to use a proper database)
        # For now, we'll store in memory
        if not hasattr(router, 'projects'):
            router.projects = {}
        router.projects[project_id] = project
        
        return {
            "success": True,
            "project_id": project_id,
            "project": project,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create project: {str(e)}")

@router.get("/project/list")
async def list_research_projects(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, description="Maximum number of projects to return"),
    offset: int = Query(0, description="Number of projects to skip")
):
    """List all research projects"""
    try:
        if not hasattr(router, 'projects'):
            router.projects = {}
        
        # Filter projects
        filtered_projects = list(router.projects.values())
        
        if status:
            filtered_projects = [p for p in filtered_projects if p["status"] == status]
        
        # Apply pagination
        total_count = len(filtered_projects)
        filtered_projects = filtered_projects[offset:offset + limit]
        
        return {
            "success": True,
            "projects": filtered_projects,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list projects: {str(e)}")

# Batch Operations
@router.post("/batch/experiment")
async def batch_experiment(
    background_tasks: BackgroundTasks,
    use_case_id: str = Form(...),
    experiment_configs: str = Form(..., description="JSON array of experiment configurations")
):
    try:
        # Check if use case exists
        if use_case_id not in experiments:
            raise HTTPException(status_code=404, detail="Use case not found")
        # Parse experiment configurations
        configs = json.loads(experiment_configs)
        if len(configs) > 10:
            raise HTTPException(status_code=400, detail="Batch size too large (max 10)")
        available_slots = RESEARCH_CONFIG["max_concurrent_experiments"] - len(active_experiments)
        if len(configs) > available_slots:
            raise HTTPException(status_code=429, detail=f"Only {available_slots} experiment slots available")
        created_experiments = []
        for i, config in enumerate(configs):
            experiment_id = str(uuid.uuid4())
            experiment = {
                "id": experiment_id,
                "use_case_id": use_case_id,
                "name": config.get("name", f"Batch Experiment {i+1}"),
                "description": config.get("description", "Batch experiment"),
                "parameters": config.get("parameters", {}),
                "dataset": config.get("dataset"),
                "status": "created",
                "created_at": datetime.now().isoformat(),
                "started_at": None,
                "completed_at": None,
                "results": None,
                "metrics": {}
            }
            experiments[use_case_id]["experiments"].append(experiment_id)
            experiment_results[experiment_id] = experiment
            created_experiments.append(experiment_id)
        for experiment_id in created_experiments:
            background_tasks.add_task(_run_experiment_background, experiment_id)
        return {
            "success": True,
            "use_case_id": use_case_id,
            "created_experiments": created_experiments,
            "total_experiments": len(created_experiments),
            "message": "Batch experiments created and started",
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create batch experiments: {str(e)}")

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
    """Get research platform system information"""
    try:
        return {
            "success": True,
            "research_available": RESEARCH_AVAILABLE,
            "moe_system_ready": get_moe_system() is not None,
            "expert_ensemble_ready": get_expert_ensemble() is not None,
            "domain_router_ready": get_domain_router() is not None,
            "cache_entries": len(_cache),
            "active_experiments": len(active_experiments),
            "total_experiments": len(experiment_results),
            "total_use_cases": len(experiments),
            "supported_features": [
                "research_assistant", "use_case_creator", "experiment_lab",
                "analysis_tools", "project_management", "batch_operations"
            ],
            "research_config": RESEARCH_CONFIG,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system info: {str(e)}") 