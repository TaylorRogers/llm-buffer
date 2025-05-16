import requests

test_api_key = "AIzaSyD9xoYuV-D5DxKfuKuMxmMrs-Q6SYnmOn8"
test_cse_id = "05927a1407a144853"  # make sure this ID is correct

params = {
    "q": "OpenAI GPT-4o",
    "key": test_api_key,
    "cx": test_cse_id,
    "num": 1
}

response = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
print("Status Code:", response.status_code)
print("Response:", response.json())