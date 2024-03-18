import json
from openai import OpenAI

MODEL = "gpt-3.5-turbo-16k"


def get_current_weather(location: str, unit="celsius") -> str:
    weather_info = {
        "location": location,
        "temperature": 25,
        "unit": "celsius",
        "forecaset": ["sunny", "windy"],
    }
    return json.dumps(weather_info)


functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state",
                },
                "celsius": {
                    "type": "string",
                    "description": "The unit of the temperature",
                },
            },
            "required": ["location"],
        },
    }
]

available_functions = {
    "get_current_weather": get_current_weather,
}


def main():
    messages = [{"role": "user", "content": "What's the weather like in Tokyo?"}]
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
    print(function_resp)

    messages.append(first_resp_msg)
    messages.append(
        {
            "role": "function",
            "name": function_call.name,
            "content": function_resp,
        }
    )
    print(messages)

    final_resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    print(final_resp)


if __name__ == "__main__":
    main()
