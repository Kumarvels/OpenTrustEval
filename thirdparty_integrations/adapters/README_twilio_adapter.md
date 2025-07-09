# Twilio SMS Adapter Integration Example

This adapter allows you to connect a Twilio SMS bot to the OpenTrustEval system for real-time answer verification and trust/hallucination detection.

## Features
- Async HTTP verification with OpenTrustEval REST API (`httpx`)
- Robust error handling and logging
- Simple response time metrics
- Customizable LLM integration point
- Clear error messages and webhook response codes

## Prerequisites
- Python 3.8+
- Install dependencies:
  ```bash
  pip install flask twilio httpx
  ```
- OpenTrustEval REST API running (default: `http://localhost:8000/thirdparty/verify_realtime`)

## Usage
1. Deploy this Flask app and set the webhook URL in your Twilio console.
2. The adapter will:
   - Receive user input from SMS
   - Call your LLM (customize `call_llm` method)
   - Verify the answer with OpenTrustEval
   - Only send trusted answers to the user

## Example Code
```python
from adapters.twilio_adapter import app
# Run with: flask run
```

## Customizing LLM Integration
Edit the `call_llm` method in `twilio_adapter.py` to connect to your LLM API. Raise exceptions on failure for robust error handling.

## Example Response Handling
- If the answer is trusted: sends the LLM answer to the user.
- If not trusted: sends a fallback message with the reason (if provided).
- If verification service is unavailable: sends a 503-style error message.

## Troubleshooting
- Ensure all dependencies are installed.
- Check logs for error details.
- Make sure the OpenTrustEval API is reachable from your bot host.

---
For more details, see the code and comments in `adapters/twilio_adapter.py`.
