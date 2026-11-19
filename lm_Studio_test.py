
import requests
import json

BASE_URL = "http://127.0.0.1:1235/v1"

MODEL = "qwen2.5-vl-3b-instruct"

print("========== LM STUDIO CONNECTION TEST ==========")

# Step 1: Check API

try:

    response = requests.get(
        f"{BASE_URL}/models",
        timeout=30
    )

    print("Models API:", response.status_code)
    print(response.text)

except Exception as e:

    print("Connection error:", str(e))
    exit()


# Step 2: Test chat completion

print("\n========== TESTING CHAT COMPLETION ==========")

payload = {

    "model": MODEL,

    "messages": [
        {
            "role": "system",
            "content": "Return valid JSON only."
        },
        {
            "role": "user",
            "content": 'Return {"status": "working"}'
        }
    ],

    "temperature": 0,

    "max_tokens": 100,

    "stream": False
}

try:

    response = requests.post(
        f"{BASE_URL}/chat/completions",
        json=payload,
        timeout=180
    )

    print("\nHTTP STATUS:", response.status_code)

    print("\n========== RAW SERVER RESPONSE ==========")

    print(response.text)

    if response.ok:

        result = response.json()

        print("\n========== MODEL OUTPUT ==========")

        print(
            result["choices"][0]["message"]["content"]
        )

    else:

        print("\nCHAT COMPLETION FAILED.")

except Exception as e:

    print("Request error:", str(e))
