"""
Adapter Example: Facebook Messenger Bot Integration
This script shows how to verify LLM answers with OpenTrustEval before responding in Messenger.

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
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.get_json()
        user_input = data['entry'][0]['messaging'][0]['message']['text']
    except Exception as e:
        logging.error(f"Invalid Messenger payload: {e}")
        return jsonify({'error': 'Invalid payload'}), 400
    try:
        candidate = call_llm(user_input)
    except Exception as e:
        logging.error(f"LLM call failed: {e}")
        return jsonify({'error': 'LLM error'}), 500
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
        response_text = "Verification service unavailable. [Code: 503]"
    else:
        if result_json.get("allow"):
            response_text = candidate
        else:
            reason = result_json.get("reason", "Untrusted answer.")
            response_text = f"Sorry, I cannot provide a reliable answer right now. [Reason: {reason}]"
    # TODO: Send response_text back to Messenger user using Facebook Send API
    logging.info(f"Responding to Messenger: {response_text}")
    return jsonify({'response': response_text}), 200

def call_llm(user_input):
    """
    Replace this method with your LLM API call.
    Raise exceptions on failure for error handling.
    """
    return "This is a generated answer."

# Usage:
# 1. Deploy this Flask app and set the webhook URL in your Facebook App settings.
# 2. Implement Facebook Send API call to reply to users.
# 3. Ensure OpenTrustEval API is running at the configured URL.
# 4. Install httpx: pip install httpx
