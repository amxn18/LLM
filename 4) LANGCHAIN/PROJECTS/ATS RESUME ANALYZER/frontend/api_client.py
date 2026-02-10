import requests

BASE_URL = "http://localhost:8000"

def call_langserve(endpoint: str, resume_text: str, jd_text: str) -> str:
    url = f"{BASE_URL}{endpoint}/invoke"

    payload = {
        "input": {
            "resume": resume_text,
            "jd": jd_text
        }
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()

    return response.json()["output"]
