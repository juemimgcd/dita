# ruff: noqa: F401  # Prepared imports for exercise bodies.
# pyright: reportUnusedImport=false
import asyncio
import logging

from langchain_core.messages import ToolMessage
from openai import APIStatusError

from dita.loop_common import RunLimitError, response_calls, validate_history
from dita.runtime_base import RunStatus, Runtime

logger = logging.getLogger(__name__)


async def run_loop(prompt: str | None, runtime: Runtime) -> str:
    status:RunStatus = "failed"
    reason = ""
    primary:BaseException | None = None
    try:
        async with asyncio.timeout(runtime.options.timeout):
            await runtime.start(prompt=prompt)
            while runtime.requests < runtime.options.max_requests:
                async with runtime.io.factory() as model:
                    await runtime.begin_turn()
                    messages = await runtime.prepare()
                    validate_history(messages)
                    try:
                        response = await runtime.io.request(
                            model=model,
                            messages=messages,
                            max_tokens=runtime.options.output_tokens
                        )
                        runtime.requests += 1
                    except APIStatusError as e:
                        if (
                            runtime.requests < runtime.options.max_requests
                            and await runtime.retry(e)
                        ):
                            continue
                        raise
                    calls = response_calls(response)
                    await runtime.on_response(response)
                    result:list[ToolMessage] = []
                    if calls:
                        if (
                            runtime.requests >= runtime.options.max_requests
                            or runtime.tool_calls + len(calls) > runtime.options.max_requests
                        ):
                            raise RuntimeError()
                        runtime.tool_calls += len(calls)
                        for call in calls:
                            execution = await runtime.execute(call)
                            await runtime.on_result(execution)
                            result.append(execution.result)
                    await runtime.after_turn(result)
                    if not calls:
                        status = "completed"
                        return response.txt or ""
            raise RunLimitError
    except BaseException as err:
        primary = err
        status = (
            "cancelled"
            if isinstance(err, asyncio.CancelledError)
            else ("stopped" if isinstance(err, RunLimitError) else "failed")
        )
        reason = (
            str(err) if isinstance(err, RunLimitError) else type(err).__name__
        )
        raise

    finally:
        try:
            await runtime.finish(status, reason)
        except BaseException as cleanup_error:
            if primary is None:
                raise
            logger.warning("cleanup failed: %s", type(cleanup_error).__name__)















































