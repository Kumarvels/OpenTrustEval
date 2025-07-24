"""
Adapter Example: Rasa Custom Action
This script shows how to verify LLM answers with OpenTrustEval in a Rasa custom action.
"""
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import requests

class ActionVerifyWithOpenTrustEval(Action):
    def name(self):
        return "action_verify_with_opentrusteval"

    def run(self, dispatcher, tracker, domain):
        user_input = tracker.latest_message.get('text')
        candidate = self.call_llm(user_input)
        result = requests.post(
            "http://localhost:8000/thirdparty/verify_realtime",
            data={"answer": candidate}
        ).json()
        if result.get("allow"):
            dispatcher.utter_message(text=candidate)
        else:
            dispatcher.utter_message(text="Sorry, I cannot provide a reliable answer right now.")
        return []

    def call_llm(self, user_input):
        # Replace with your LLM API call
        return "This is a generated answer."
