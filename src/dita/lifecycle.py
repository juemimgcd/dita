# ruff: noqa: F401  # Prepared imports for exercise bodies.
# pyright: reportUnusedImport=false
import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Event:
    kind: str
    detail: str = ""
    call_id: str = ""


type Listener = Callable[[Event], Awaitable[None]]


async def emit(event: Event, listeners: Sequence[Listener]) -> None:
    for listener in listeners:
        try:
            await listener(event)
        except Exception as e:
            logger.warning("event listener failed: %s", type(e).__name__)


async def finish_event(status: str, listeners: Sequence[Listener]) -> None:
    try:
        async with asyncio.timeout(1.0):
            await emit(Event("run_end", status), listeners)
    except TimeoutError:
        logger.warning("run_end listeners timed out")
