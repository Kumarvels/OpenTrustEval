"""
Example: Cloudscale REST API Endpoint (FastAPI)
Best practice: Separate business logic, use async, validate input, and document with OpenAPI.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class TrustEvalRequest(BaseModel):
    text: str = None
    image_url: str = None
    image_model: str = 'EfficientNetB0'

@router.post("/evaluate", summary="Evaluate trust for text/image input")
async def evaluate(request: TrustEvalRequest):
    # TODO: Call business logic, validate input, handle errors
    if not request.text and not request.image_url:
        raise HTTPException(status_code=400, detail="At least one of text or image_url required.")
    # result = run_pipeline(request.text, request.image_url, request.image_model)
    return {"result": "stub", "input": request.dict()}
