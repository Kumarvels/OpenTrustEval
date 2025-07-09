"""
Adapter Example: Slack Bot Integration
This script shows how to verify LLM answers with OpenTrustEval before posting to Slack.
"""
import os
import requests
from slack_bolt import App

slack_app = App(token=os.environ["SLACK_BOT_TOKEN"])

@slack_app.event("message")
def handle_message_events(body, say):
    user_input = body['event']['text']
    candidate = call_llm(user_input)
    result = requests.post(
        "http://localhost:8000/thirdparty/verify_realtime",
        data={"answer": candidate}
    ).json()
    if result.get("allow"):
        say(candidate)
    else:
        say("Sorry, I cannot provide a reliable answer right now.")

def call_llm(user_input):
    # Replace with your LLM API call
    return "This is a generated answer."
