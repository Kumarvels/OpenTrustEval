"""
Adapter Example: LLM Client
This script shows how to call a 3rd-party LLM, verify its answer with OpenTrustEval, and only respond if allowed.
"""
import requests

def get_llm_answer(user_input):
    # Replace with your LLM API call
    return "This is a generated answer."

def verify_with_opentrusteval(answer):
    resp = requests.post(
        "http://localhost:8000/thirdparty/verify_realtime",
        data={"answer": answer}
    )
    return resp.json()

def main():
    user_input = input("User: ")
    candidate = get_llm_answer(user_input)
    result = verify_with_opentrusteval(candidate)
    if result.get("allow"):
        print("Bot:", candidate)
    else:
        print("Bot: Sorry, I cannot provide a reliable answer right now.")

if __name__ == "__main__":
    main()
