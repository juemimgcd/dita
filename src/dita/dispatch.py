import asyncio
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

from langchain_core.messages import ToolCall, ToolMessage

from dita.hooks import Decision, Hooks, ToolContext
from dita.lifecycle import Event, Listener, emit
from dita.runtime_base import ToolExecution
from dita.tools import ToolError, make_tool_message, resolve_tool_call


async def execute_tool(
        call: ToolCall,
        workspace: Path,
        hooks: Hooks,
        listeners: Sequence[Listener],
) -> ToolExecution:
    raw: ToolMessage
    try:
        handler,args = resolve_tool_call(call)
    except ToolError as e:
        raw = make_tool_message(call,str(e),"failed")
    else:
        decision = await hooks.invoke(
            "before_model",
            ToolContext(call["name"],(call["id"] or ""),args.model_dump(mode="json"))
        )
        if not isinstance(decision,Decision):
            raise TypeError
        if decision.stop:
            raw = make_tool_message(call,str(decision.reason),"denied")
        else:
            await emit(Event("tool_start", call["name"], (call["id"] or "")), listeners)
            try:
                content = await asyncio.to_thread(handler,args,workspace)
            except ToolError as e:
                raw = make_tool_message(call,str(e),"failed")
            else:
                raw = make_tool_message(call,str(content),"success")
    result = await hooks.invoke(
        "after_tool",
        raw
    )
    if not isinstance(result,ToolMessage):
        raise TypeError
    return ToolExecution(deepcopy(raw),deepcopy(result))



