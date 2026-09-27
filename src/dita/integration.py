# ruff: noqa: F401  # Prepared imports for exercise bodies.
# pyright: reportUnusedImport=false
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.messages import BaseMessage
from openai import APIStatusError

from dita.compaction import choose_compaction_range, compact
from dita.context import (
    Budget,
    Resource,
    Summary,
    build_context,
    estimate_tokens,
    uncovered_entries,
)
from dita.hooks import Hooks
from dita.lifecycle import Listener
from dita.loop import run_loop
from dita.loop_common import INSTRUCTIONS, RunLimitError, validate_history
from dita.memory import Memory, MemoryStore, recall
from dita.model_io import request_once, summarize_once
from dita.runtime_base import ModelIO, RequestFn, RunOptions, SummaryFn
from dita.session import load_session, save_session
from dita.session_runtime import SessionRuntime
from dita.storage import JsonStore


@dataclass
class Services:
    database: JsonStore
    memories: MemoryStore
    workspace: Path
    budget: Budget = field(default_factory=Budget)
    hooks: Hooks = field(default_factory=Hooks)
    listeners: Sequence[Listener] = ()
    resources: Sequence[Resource] = ()
    recall_query: str = ""
    request: RequestFn = request_once
    summarizer: SummaryFn = summarize_once


def read_summary(services: Services, session_id: str) -> Summary | None:
    body = services.database.get("summary", session_id)
    if not body:
        return None
    summary = Summary.model_validate_json(body)
    for memory_id in summary.memory_ids:
        record = services.database.get("memory", memory_id)
        memory = None if record is None else Memory.model_validate_json(record)
        if (
                memory is None
                or not memory.active
                or memory.scope not in services.memories.allowed_scopes
        ):
            raise ValueError
    return summary


class ContextRuntime(SessionRuntime):
    def __init__(self, session_id: str, services: Services) -> None:
        super().__init__(
            services.database,
            session_id,
            services.workspace,
            allowed_scopes=services.memories.allowed_scopes,
            hooks=services.hooks,
            listeners=services.listeners,
            io=ModelIO(request=services.request, summarize=services.summarizer),
            options=RunOptions(output_tokens=services.budget.output),
        )
        self.services = services
        self.query = services.recall_query
        self.force_compaction = False
        self.overflow_retried = False

    async def start(self, prompt: str | None) -> None:
        await super().start(prompt)
        self.query = self.services.recall_query or prompt or ""

    async def prepare(self) -> list[BaseMessage]:
        services = self.services
        session = load_session(self.database,self.session_id)
        memories = recall(
            services.memories,
            self.query,
            self.allowed_scopes,
        )
        with self.database.transaction():
            session.memory_ids = sorted(
                set(session.memory_ids) | {memory.id for memory in memories},
            )
            save_session(self.database,session)
        summary = read_summary(services, self.session_id)
        view = build_context(
            INSTRUCTIONS,
            services.resources,
            memories,
            summary,
            session.entries,
            services.budget,
        )
        old_tasks = choose_compaction_range(uncovered_entries(session.entries,summary))
        proactive = bool(old_tasks) and view.estimated_tokens >= int(services.budget.input_limit*0.8)
        if self.force_compaction or view.needs_compaction or proactive:
            candidate = await compact(
                session.entries,
                summary,
                self.io.summarize,
                budget=services.budget,
                memory_ids=session.memory_ids,
            )
            candidate_view = build_context(
                INSTRUCTIONS,
                services.resources,
                memories,
                candidate,
                session.entries,
                services.budget,
            )
            if (
                    candidate_view.needs_compaction
                    or candidate_view.estimated_tokens >= view.estimated_tokens
                    or candidate_view.estimated_tokens
                    > int(services.budget.input_limit * 0.65)
            ):
                raise ValueError("compaction did not meet the context target")

            with self.database.transaction():
                self.database.put(
                    "summary",self.session_id,candidate.model_dump_json()
                )

        request_view = await self.apply_before_model(view.messages)
        validate_history(request_view)
        if estimate_tokens(request_view) > services.budget.input_limit:
            raise RunLimitError("before_model exceeded the input allowance")
        return request_view



    async def retry(self, error: APIStatusError) -> bool:
        detail = str(error.body).casefold()
        overflow = error.status_code == 400 and any(
            marker in detail
            for marker in (
                "context_length_exceeded",
                "maximum context length",
                "context window",
            )
        )
        if not overflow or self.overflow_retried:
            return False
        self.overflow_retried = self.force_compaction = True
        return True


async def run_session_task(
        session_id: str,
        prompt: str | None,
        services: Services,
) -> str:
    runtime = ContextRuntime(session_id, services)
    return await run_loop(prompt, runtime)


























    async def retry(self, error: APIStatusError) -> bool:
        detail = str(error.body).casefold()
        overflow = error.status_code == 400 and any(
            marker in detail
            for marker in (
                "context_length_exceeded",
                "maximum context length",
                "context window",
            )
        )
        if not overflow or self.overflow_retried:
            return False
        self.overflow_retried = self.force_compaction = True
        return True


async def run_session_task(
        session_id: str,
        prompt: str | None,
        services: Services,
) -> str:
    runtime = ContextRuntime(session_id, services)
    return await run_loop(prompt, runtime)
