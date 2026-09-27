import os
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from dita.loop_common import ModelResponseError, response_calls
from dita.tools import ToolSchema, tool_schemas


@asynccontextmanager
async def create_model() -> AsyncGenerator[ChatOpenAI]:
    with httpx.Client() as client:
        async with httpx.AsyncClient() as session:
            yield ChatOpenAI(
                model=os.environ["LLM_MODEL"],
                base_url=os.environ["LLM_BASE_URL"],
                api_key=SecretStr(os.environ["LLM_API_KEY"]),
                use_responses_api=False,
                timeout=60.0,
                max_retries=0,
                cache=False,
                http_client=client,
                http_async_client=session,
            )


async def request_once(
        model: ChatOpenAI,
        history: Sequence[BaseMessage],
        *,
        tools: Sequence[ToolSchema] | None = None,
        max_tokens: int = 2048
) -> AIMessage:
    schemas = tool_schemas() if tools is not None else tool_schemas()
    requester = model.bind_tools(schemas) if schemas else model
    return await requester.ainvoke(list(history), max_tokens=max_tokens)


async def summarize_once(
        prompt: str
) -> str:
    async with create_model() as model:
        response = await request_once(model, [HumanMessage(content=prompt)],tools=())
    if response_calls(response):
        raise ModelResponseError
    return response.text
