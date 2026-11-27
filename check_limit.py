import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load your GROQ_API_KEY from .env
load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# 2. Send a tiny 1-word test message and ask for the "raw" response (the receipt)
raw_response = client.chat.completions.with_raw_response.create(
    model="qwen/qwen3.8-27b",
    messages=[{"role": "user", "content": "hi"}],
    max_tokens=1,
    reasoning_effort="none",
)

# 3. Read the numbers from Groq's receipt
headers = raw_response.headers

print("--- GROQ LIVE QUOTA RECEIPT ---")
print(
    f"Daily Requests Left : {headers.get('x-ratelimit-remaining-requests')} / {headers.get('x-ratelimit-limit-requests')}"
)
print(f"Daily Reset In      : {headers.get('x-ratelimit-reset-requests')}")
print("-" * 31)
print(
    f"Minute Tokens Left  : {headers.get('x-ratelimit-remaining-tokens')} / {headers.get('x-ratelimit-limit-tokens')}"
)
print(f"Minute Reset In     : {headers.get('x-ratelimit-reset-tokens')}")