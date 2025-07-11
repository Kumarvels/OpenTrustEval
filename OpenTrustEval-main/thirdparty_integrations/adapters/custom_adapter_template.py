"""
Custom Adapter Template
Use this template to build integrations for any other platform or business system.
"""
import requests

def get_user_input():
    # Replace with code to get user input from your platform
    return "User's question"

def send_response(response_text):
    # Replace with code to send response back to your platform
    print("Bot:", response_text)

def call_llm(user_input):
    # Replace with your LLM API call
    return "This is a generated answer."

def main():
    user_input = get_user_input()
    candidate = call_llm(user_input)
    result = requests.post(
        "http://localhost:8000/thirdparty/verify_realtime",
        data={"answer": candidate}
    ).json()
    if result.get("allow"):
        send_response(candidate)
    else:
        send_response("Sorry, I cannot provide a reliable answer right now.")

if __name__ == "__main__":
    main()
