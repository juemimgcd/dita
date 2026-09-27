from pathlib import Path

from dita.loop import run_loop
from dita.runtime_base import Runtime


async def run_agent(
        prompt:str,
        workspace:Path,
        *,
        runtime:Runtime | None = None,
) -> str:
    active = runtime if runtime is not None else Runtime(workspace)
    if active.workspace != workspace.resolve(strict=True):
        raise ValueError(
        )
    return await run_loop(prompt,active)

