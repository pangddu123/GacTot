import requests

if __name__ == "__main__":

    url = "http://0.0.0.0:8000/api/generate/"

    data = {
        "messages_list": [

            [{"role": "user", "content": "who are you"}],
        ],
        "max_new_tokens": 1024,
        "apply_chat_template": True,
    }

    response = requests.post(url, json=data)
    print(response.json())
