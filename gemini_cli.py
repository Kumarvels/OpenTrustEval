# gemini_cli.py
import google.generativeai as genai
import sys
import os

# Set the API key here
GEMINI_API_KEY = "AIzaSyBFk7jt2ahQjHlMaUMosyx01Lci-2-7hjg"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

prompt = " ".join(sys.argv[1:])
response = model.generate_content(prompt)
print(response.text)
