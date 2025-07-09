"""
Adapter Example: Twilio SMS Integration
This script shows how to verify LLM answers with OpenTrustEval before sending SMS replies.

Improvements:
- Robust error handling for HTTP and LLM errors
- Logging for debugging
- Clear extension points for LLM integration
- Usage instructions
- Async HTTP verification (httpx)
- Simple metrics (response time logging)
- Webhook response code and improved error messages
"""
import logging
import time
import httpx
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

@app.route('/sms', methods=['POST'])
def sms_reply():
    try:
        user_input = request.form['Body']
    except Exception as e:
        logging.error(f"Invalid Twilio payload: {e}")
        resp = MessagingResponse()
        resp.message("Invalid request.")
        return str(resp), 400
    try:
        candidate = call_llm(user_input)
    except Exception as e:
        logging.error(f"LLM call failed: {e}")
        resp = MessagingResponse()
        resp.message("Error generating answer.")
        return str(resp), 500
    start_time = time.time()
    try:
        # Async HTTP call to OpenTrustEval
        import asyncio
        async def verify():
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.post(
                    "http://localhost:8000/thirdparty/verify_realtime",
                    data={"answer": candidate}
                )
                response.raise_for_status()
                return response.json()
        result_json = asyncio.run(verify())
        elapsed = time.time() - start_time
        logging.info(f"Verification API response time: {elapsed:.3f}s")
    except Exception as e:
        logging.error(f"Verification API call failed: {e}")
        resp = MessagingResponse()
        resp.message("Verification service unavailable. [Code: 503]")
        return str(resp), 503
    resp = MessagingResponse()
    if result_json.get("allow"):
        resp.message(candidate)
    else:
        reason = result_json.get("reason", "Untrusted answer.")
        resp.message(f"Sorry, I cannot provide a reliable answer right now. [Reason: {reason}]")
    return str(resp)

def call_llm(user_input):
    """
    Replace this method with your LLM API call.
    Raise exceptions on failure for error handling.
    """
    return "This is a generated answer."

# Usage:
# 1. Deploy this Flask app and set the webhook URL in your Twilio console.
# 2. Ensure OpenTrustEval API is running at the configured URL.
# 3. Test by sending an SMS to your Twilio number.
# 4. Install httpx: pip install httpx
