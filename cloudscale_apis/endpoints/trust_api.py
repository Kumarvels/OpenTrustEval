# src/api/trust_api.py
"""
Trust API and Microservices - Scalable trust evaluation services
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import asyncio
import uuid
from datetime import datetime
import redis
import json
import logging

logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="OpenTrustEval API",
    description="Scalable trust evaluation API with microservices architecture",
    version="1.0.0"
)

# Redis for caching and job queue
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Pydantic models
class TrustEvaluationRequest(BaseModel):
    model_id: str = Field(..., description="Unique identifier for the model")
    model_type: str = Field(default="llm", description="Type of model (llm, ml, cv, etc.)")
    data: Dict[str, Any] = Field(..., description="Evaluation data")
    evaluation_config: Optional[Dict[str, Any]] = Field(default={}, description="Evaluation configuration")
    callback_url: Optional[str] = Field(default=None, description="URL for callback notifications")

class TrustEvaluationResponse(BaseModel):
    evaluation_id: str
    status: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

class TrustEvaluationStatus(BaseModel):
    evaluation_id: str
    status: str
    progress: Optional[float] = None
    estimated_completion: Optional[datetime] = None

# In-memory storage for demonstration (use database in production)
evaluation_storage = {}

# Trust evaluation service
class TrustEvaluationService:
    """Service for trust evaluation operations"""
    
    @staticmethod
    async def execute_evaluation(request: TrustEvaluationRequest) -> Dict[str, Any]:
        """Execute trust evaluation"""
        try:
            # Import evaluator (lazy import for performance)
            from src.evaluators.composite_evaluator import CompositeTrustEvaluator
            
            # Create evaluator
            evaluator = CompositeTrustEvaluator()
            
            # Execute evaluation
            results = evaluator.evaluate_comprehensive_trust(
                model=None,  # In real implementation, load model by model_id
                data=request.data,
                model_type=request.model_type,
                **request.evaluation_config
            )
            
            return {
                'status': 'completed',
                'results': results,
                'completed_at': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'completed_at': datetime.now()
            }
    
    @staticmethod
    async def queue_evaluation(request: TrustEvaluationRequest) -> str:
        """Queue evaluation for background processing"""
        evaluation_id = str(uuid.uuid4())
        
        # Store initial request
        evaluation_storage[evaluation_id] = {
            'request': request.dict(),
            'status': 'queued',
            'created_at': datetime.now()
        }
        
        # Add to Redis queue
        job_data = {
            'evaluation_id': evaluation_id,
            'request': request.dict()
        }
        redis_client.lpush('trust_evaluation_queue', json.dumps(job_data))
        
        return evaluation_id

# API endpoints
@app.post("/evaluate", response_model=TrustEvaluationResponse)
async def evaluate_trust(request: TrustEvaluationRequest, background_tasks: BackgroundTasks):
    """Submit trust evaluation request"""
    try:
        # Queue for background processing
        evaluation_id = await TrustEvaluationService.queue_evaluation(request)
        
        return TrustEvaluationResponse(
            evaluation_id=evaluation_id,
            status="queued",
            created_at=evaluation_storage[evaluation_id]['created_at']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluation/{evaluation_id}", response_model=TrustEvaluationResponse)
async def get_evaluation_status(evaluation_id: str):
    """Get evaluation status and results"""
    if evaluation_id not in evaluation_storage:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    stored_data = evaluation_storage[evaluation_id]
    
    return TrustEvaluationResponse(
        evaluation_id=evaluation_id,
        status=stored_data['status'],
        results=stored_data.get('results'),
        error=stored_data.get('error'),
        created_at=stored_data['created_at'],
        completed_at=stored_data.get('completed_at')
    )

@app.get("/evaluation/{evaluation_id}/status", response_model=TrustEvaluationStatus)
async def get_evaluation_progress(evaluation_id: str):
    """Get evaluation progress"""
    if evaluation_id not in evaluation_storage:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    stored_data = evaluation_storage[evaluation_id]
    
    return TrustEvaluationStatus(
        evaluation_id=evaluation_id,
        status=stored_data['status'],
        progress=stored_data.get('progress', 0.0) if stored_data['status'] == 'processing' else 
                (1.0 if stored_data['status'] == 'completed' else 0.0)
    )

# Background worker
async def trust_evaluation_worker():
    """Background worker for processing trust evaluations"""
    logger.info("Starting trust evaluation worker")
    
    while True:
        try:
            # Get job from Redis queue
            job_data_json = redis_client.brpop('trust_evaluation_queue', timeout=1)
            
            if job_data_json:
                _, job_data_bytes = job_data_json
                job_data = json.loads(job_data_bytes.decode('utf-8'))
                
                evaluation_id = job_data['evaluation_id']
                request_data = job_data['request']
                
                logger.info(f"Processing evaluation {evaluation_id}")
                
                # Update status
                evaluation_storage[evaluation_id]['status'] = 'processing'
                evaluation_storage[evaluation_id]['progress'] = 0.1
                
                # Execute evaluation
                request = TrustEvaluationRequest(**request_data)
                results = await TrustEvaluationService.execute_evaluation(request)
                
                # Update storage
                evaluation_storage[evaluation_id].update(results)
                evaluation_storage[evaluation_id]['status'] = results['status']
                
                # Trigger callback if provided
                if request.callback_url:
                    await trigger_callback(request.callback_url, evaluation_id, results)
                
                logger.info(f"Completed evaluation {evaluation_id}")
            
            await asyncio.sleep(0.1)  # Prevent busy waiting
            
        except Exception as e:
            logger.error(f"Worker error: {e}")
            await asyncio.sleep(1)  # Slow down on errors

async def trigger_callback(callback_url: str, evaluation_id: str, results: Dict[str, Any]):
    """Trigger callback notification"""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            callback_data = {
                'evaluation_id': evaluation_id,
                'results': results
            }
            await client.post(callback_url, json=callback_data)
    except Exception as e:
        logger.error(f"Callback failed: {e}")

# Batch evaluation endpoints
class BatchEvaluationRequest(BaseModel):
    evaluations: List[TrustEvaluationRequest]
    batch_config: Optional[Dict[str, Any]] = Field(default={}, description="Batch processing configuration")

class BatchEvaluationResponse(BaseModel):
    batch_id: str
    status: str
    completed_evaluations: int = 0
    total_evaluations: int
    results: Optional[List[Dict[str, Any]]] = None

@app.post("/batch-evaluate", response_model=BatchEvaluationResponse)
async def batch_evaluate(request: BatchEvaluationRequest, background_tasks: BackgroundTasks):
    """Submit batch trust evaluation request"""
    batch_id = str(uuid.uuid4())
    
    # Queue individual evaluations
    evaluation_ids = []
    for eval_request in request.evaluations:
        eval_id = await TrustEvaluationService.queue_evaluation(eval_request)
        evaluation_ids.append(eval_id)
    
    # Store batch information
    batch_storage[batch_id] = {
        'evaluation_ids': evaluation_ids,
        'status': 'processing',
        'total_evaluations': len(evaluation_ids),
        'completed_evaluations': 0,
        'created_at': datetime.now()
    }
    
    # Start batch monitoring in background
    background_tasks.add_task(monitor_batch_progress, batch_id, evaluation_ids)
    
    return BatchEvaluationResponse(
        batch_id=batch_id,
        status="processing",
        total_evaluations=len(evaluation_ids)
    )

# Batch storage
batch_storage = {}

async def monitor_batch_progress(batch_id: str, evaluation_ids: List[str]):
    """Monitor batch evaluation progress"""
    while batch_storage[batch_id]['completed_evaluations'] < batch_storage[batch_id]['total_evaluations']:
        completed_count = 0
        for eval_id in evaluation_ids:
            if eval_id in evaluation_storage:
                if evaluation_storage[eval_id]['status'] in ['completed', 'failed']:
                    completed_count += 1
        
        batch_storage[batch_id]['completed_evaluations'] = completed_count
        
        if completed_count == len(evaluation_ids):
            batch_storage[batch_id]['status'] = 'completed'
            # Collect results
            results = []
            for eval_id in evaluation_ids:
                if eval_id in evaluation_storage:
                    results.append(evaluation_storage[eval_id])
            batch_storage[batch_id]['results'] = results
            break
        
        await asyncio.sleep(5)  # Check every 5 seconds

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "running",
            "worker": "running" if is_worker_alive() else "stopped",
            "redis": "connected" if is_redis_connected() else "disconnected"
        }
    }

def is_worker_alive() -> bool:
    """Check if worker is alive"""
    # Implementation would check worker status
    return True

def is_redis_connected() -> bool:
    """Check Redis connection"""
    try:
        redis_client.ping()
        return True
    except:
        return False

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("Starting Trust API service")
    
    # Start background worker
    asyncio.create_task(trust_evaluation_worker())
    
    logger.info("Trust API service started")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("Shutting down Trust API service")

# CLI for API management
def start_api_server():
    """Start API server"""
    import uvicorn
    uvicorn.run(
        "src.api.trust_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    start_api_server()
