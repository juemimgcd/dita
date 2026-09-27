import asyncio

history = ["hello"]
max_requests = 8
max_tool_calls = 12
timeout = 120
budget=512

def prepare_contex(history,budget):
    pass

def validate(messages):
    pass

def extract_too_calls(messages):
    pass

def request_once(messages):
    pass

def final_text():
    pass

def execute_tool(calls):
    pass

tool_calls = 0

try:
    with asyncio.timeout(timeout):
        for step in range(max_requests):
            messages = prepare_contex(history,budget)
            validate(messages)
            try:
                response = request_once(messages)
            except asyncio.TimeoutError:
                if step < max_requests:
                    continue
            calls = extract_too_calls(messages)
            if not calls:
                return final_text(history)
            if step < max_requests and tool_calls < max_tool_calls:
                for call in calls:
                    result = execute_tool(call)
                    history.append(result)
                    tool_calls+=1

        raise RuntimeError



































































