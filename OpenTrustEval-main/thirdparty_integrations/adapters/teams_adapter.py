"""
Adapter Example: Microsoft Teams Bot Integration
This script shows how to verify LLM answers with OpenTrustEval before responding in Teams.

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
from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import Activity, ActivityTypes

logging.basicConfig(level=logging.INFO)

class TeamsBot(ActivityHandler):
    async def on_message_activity(self, turn_context: TurnContext):
        user_input = turn_context.activity.text
        try:
            candidate = self.call_llm(user_input)
        except Exception as e:
            logging.error(f"LLM call failed: {e}")
            await turn_context.send_activity(Activity(type=ActivityTypes.message, text="Error generating answer."))
            return
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.post(
                    "http://localhost:8000/thirdparty/verify_realtime",
                    data={"answer": candidate}
                )
                response.raise_for_status()
                result_json = response.json()
            elapsed = time.time() - start_time
            logging.info(f"Verification API response time: {elapsed:.3f}s")
        except Exception as e:
            logging.error(f"Verification API call failed: {e}")
            await turn_context.send_activity(Activity(type=ActivityTypes.message, text="Verification service unavailable. [Code: 503]"))
            return
        if result_json.get("allow"):
            await turn_context.send_activity(Activity(type=ActivityTypes.message, text=candidate))
        else:
            reason = result_json.get("reason", "Untrusted answer.")
            await turn_context.send_activity(Activity(type=ActivityTypes.message, text=f"Sorry, I cannot provide a reliable answer right now. [Reason: {reason}]") )

    def call_llm(self, user_input):
        """
        Replace this method with your LLM API call.
        Raise exceptions on failure for error handling.
        """
        # Example: raise Exception("LLM not available")
        return "This is a generated answer."

# Usage:
# 1. Register TeamsBot with your Microsoft Bot Framework app.
# 2. Ensure OpenTrustEval API is running at the configured URL.
# 3. Deploy to Azure Bot Service or your preferred Teams bot hosting.
# 4. Install httpx: pip install httpx
