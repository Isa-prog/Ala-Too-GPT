import json
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored

GPT_MODEL = "gpt-4o"
client = OpenAI()

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


def pretty_print_conversation(messages):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }

    for message in messages:
        if message["role"] == "system":
            print(colored(f"system: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "user":
            print(colored(f"user: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and message.get("function_call"):
            print(colored(f"assistant: {message['function_call']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and not message.get("function_call"):
            print(colored(f"assistant: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "function":
            print(colored(f"function ({message['name']}): {message['content']}\n", role_to_color[message["role"]]))


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": "Get today's weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                },
                "required": ["location", "format"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_n_day_weather_forecast",
            "description": "Get an N-day weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use. Infer this from the users location.",
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "The number of days to forecast",
                    }
                },
                "required": ["location", "format", "num_days"]
            },
        }
    },
]

messages = []
messages.append({"role": "system",
                 "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})


def call_function(function_name, parameters):
    if function_name == "get_weather_forecast":
        return json.dumps({
            "location": "San Francisco, CA",
            "format": "celsius"
        })
    elif function_name == "get_n_day_weather_forecast":
        return json.dumps({
            "location": "San Francisco, CA",
            "format": "celsius",
            "num_days": 3
        })
    else:
        return None


while True:
    user_message = input("You: ")
    if user_message == "exit":
        break

    messages.append({"role": "user", "content": user_message})
    chat_response = chat_completion_request(
        messages, tools=tools
    )

    assistant_message = chat_response.choices[0].message
    messages.append(assistant_message)

    if assistant_message.get("function_call"):
        function_name = assistant_message["function_call"]["name"]
        parameters = assistant_message["function_call"]["arguments"]
        function_response = call_function(function_name, json.loads(parameters))

        messages.append({
            "role": "function",
            "name": function_name,
            "content": function_response,
        })

        chat_response = chat_completion_request(
            messages, tools=tools
        )
        assistant_message = chat_response.choices[0].message
        messages.append(assistant_message)

    pretty_print_conversation(messages)
