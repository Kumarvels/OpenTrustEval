"""
Example: Webhook Receiver (FastAPI)
Best practice: Validate payload, log events, and respond quickly (async).
"""
from fastapi import APIRouter, Request, HTTPException
import logging

router = APIRouter()

@router.post("/webhook/event", summary="Receive event webhook")
async def receive_event(request: Request):
    try:
        payload = await request.json()
        # TODO: Validate and process payload
        logging.info(f"Received webhook: {payload}")
        return {"status": "received"}
    except Exception as e:
        logging.error(f"Webhook error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
