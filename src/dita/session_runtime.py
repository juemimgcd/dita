from collections.abc import Sequence
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from dita.hook_runtime import HookRuntime
from dita.hooks import Hooks
from dita.lifecycle import Event, Listener, emit
from dita.runtime_base import ModelIO, RunOptions, RunStatus, ToolExecution
from dita.session import (
    append_user,
    commit_response,
    commit_tool_result,
    decode,
    finish_session,
    history_messages,
    load_session,
    save_session,
)
from dita.storage import JsonStore


class SessionRuntime(HookRuntime):
    """Supplied adapter: persistence precedes HookRuntime's observation hooks."""

    def __init__(
            self,
            database: JsonStore,
            session_id: str,
            workspace: Path,
            *,
            allowed_scopes: frozenset[str],
            hooks: Hooks | None = None,
            listeners: Sequence[Listener] = (),
            io: ModelIO | None = None,
            options: RunOptions | None = None,
    ) -> None:
        super().__init__(
            workspace, hooks=hooks, listeners=listeners, io=io, options=options
        )
        self.database = database
        self.session_id = session_id
        self.allowed_scopes = allowed_scopes
        self.owns_run = False

    async def start(self, prompt: str | None) -> None:
        if self.started:
            raise ValueError("create a fresh Runtime to resume")
        session = load_session(self.database,self.session_id)
        if session.scope not in self.allowed_scopes:
            raise PermissionError(f"scope {self.allowed_scopes} not allowed")
        if session.pending:
            raise ValueError("session pending is allowed")
        if prompt is not None:
            append_user(self.database,self.session_id,prompt)
        else:
            if not session.entries or not isinstance(decode(session.entries[-1]),(HumanMessage,ToolMessage)):
                raise ValueError
            with self.database.transaction():
                session.status = "running"
                session.reason = ""
                save_session(self.database,session)
        self.started = self.owns_run = True
        self.history = history_messages(
            load_session(self.database,self.session_id).entries
        )
        await emit(Event("run_start"), self.listeners)


    async def on_response(self, response: AIMessage) -> None:
        commit_response(self.database, self.session_id, response)
        await super().on_response(response)

    async def on_result(self, execution: ToolExecution) -> None:
        commit_tool_result(
            self.database, self.session_id, execution.result, execution.raw
        )
        await super().on_result(execution)

    async def finish(self, status: RunStatus, reason: str) -> None:
        try:
            if self.owns_run:
                finish_session(self.database, self.session_id, status, reason)
        finally:
            await super().finish(status, reason)




