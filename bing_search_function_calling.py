import json
from openai import OpenAI
import requests
import os
import json

BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search"
MODEL = "gpt-3.5-turbo-16k"
COUNT = 3

# https://api.python.langchain.com/en/latest/_modules/langchain_community/utilities/bing_search.html#BingSearchAPIWrapper


def bing_search(count: int, q: str) -> str:
    headers = {"Ocp-Apim-Subscription-Key": os.environ["BING_SEARCH_API_KEY"]}
    params = {
        "count": count,
        "q": q,
        "textDecorations": True,
        "textFormat": "HTML",
    }
    response = requests.get(BING_SEARCH_URL, headers=headers, params=params)
    response.raise_for_status()
    raw_results = response.json()
    search_results = []
    for result in raw_results["webPages"]["value"]:
        search_result = {
            "name": result["name"],
            "url": result["url"],
            "snippet": result["snippet"],
        }
        search_results.append(search_result)
    return json.dumps(search_results)


functions = [
    {
        "name": "bing_search",
        "description": "Search the term using Bing Search Web API",
        "parameters": {
            "type": "object",
            "properties": {
                "count": {
                    "type": "number",
                    "description": "The number of search results to return in the response. The default is 10 and the maximum value is 50.",
                },
                "q": {
                    "type": "string",
                    "description": "The user's search query term. The term may not be empty.",
                },
            },
            "required": ["count", "search_term"],
        },
    }
]

available_functions = {"bing_search": bing_search}


def main():
    messages = [
        {
            "role": "user",
            "content": "Tell me the latest product which Apple published in 2024. Tell me with the source URL.",
        }
    ]
    client = OpenAI()
    first_resp = client.chat.completions.create(
        functions=functions,
        messages=messages,
        model=MODEL,
    )
    print(first_resp)

    first_resp_msg = first_resp.choices[0].message
    function_call = first_resp_msg.function_call
    function = available_functions[function_call.name]
    kwargs = json.loads(function_call.arguments)
    function_resp = function(**kwargs)
    # print(function_resp)

    messages.append(first_resp_msg)
    messages.append(
        {
            "role": "function",
            "name": function_call.name,
            "content": function_resp,
        }
    )
    # print(messages)

    final_resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    print(final_resp)


if __name__ == "__main__":
    main()
