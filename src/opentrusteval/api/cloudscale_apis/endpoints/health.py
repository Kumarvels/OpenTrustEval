"""
Health Check Endpoint Example (FastAPI)
"""
from fastapi import APIRouter

router = APIRouter()

@router.get("/health", summary="Health check")
def health():
    return {"status": "ok"}
