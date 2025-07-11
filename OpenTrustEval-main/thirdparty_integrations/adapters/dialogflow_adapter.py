"""
Adapter Example: Google Dialogflow Integration
This script shows how to use a webhook fulfillment to verify LLM answers with OpenTrustEval before responding.
"""
from fastapi import FastAPI, Request
import requests

app = FastAPI()

@app.post("/dialogflow/webhook")
async def dialogflow_webhook(request: Request):
    data = await request.json()
    user_input = data['queryResult']['queryText']
    candidate = call_llm(user_input)  # Replace with your LLM call
    result = requests.post(
        "http://localhost:8000/thirdparty/verify_realtime",
        data={"answer": candidate}
    ).json()
    if result.get("allow"):
        response_text = candidate
    else:
        response_text = "Sorry, I cannot provide a reliable answer right now."
    return {"fulfillmentText": response_text}

def call_llm(user_input):
    # Replace with your LLM API call
    return "This is a generated answer."
