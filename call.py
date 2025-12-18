import requests

if __name__ == "__main__":

    url = "http://0.0.0.0:8000/api/generate/"

    data = {
        "messages_list": [

            [{"role": "user", "content": "Which is greater, 9.9 or 9.11?"}],
        ],
        "max_new_tokens": 500,
        "apply_chat_template": True,
    }

    response = requests.post(url, json=data)
    print(response.json())
