import requests

def query(payload, model_id, API_TOKEN):
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()