# pyright: reportUnusedImport=false
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from dita.session import SessionData
from dita.storage import JsonStore


class Memory(BaseModel):
    # 记忆的唯一标识，也是持久化保存和读取记忆时使用的键。
    id: str = Field(default_factory=lambda: uuid4().hex)
    # 记忆的正文内容；remember 写入时会去掉首尾空白。
    text: str
    # 记忆所属的访问范围，写入时必须属于 MemoryStore 的可信范围。
    scope: str
    # 来源会话中某条 Entry 的 ID，用于追溯这条记忆的原始消息。
    source_entry_id: str
    # 记忆的创建时间，使用 UTC 时区的 ISO 8601 字符串。
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    # 记忆是否有效；False 表示逻辑失效，forget 不删除原始记录。
    active: bool = True


class MemoryStore:
    def __init__(self, database: JsonStore, allowed_scopes: frozenset[str]) -> None:
        if not allowed_scopes:
            raise ValueError("at least one trusted scope is required")
        # 保存会话和记忆的底层 JSON 存储，同时提供事务支持。
        self.database = database
        # 调用方传入的可信访问范围集合，用于限制记忆写入和来源校验。
        self.allowed_scopes = allowed_scopes


def remember(
        store: MemoryStore, text: str, scope: str, source_entry_id: str, *, confirmed: bool
) -> Memory:
    if not confirmed:
        raise PermissionError("explicit confirmation required")
    if scope not in store.allowed_scopes:
        raise ValueError(f"scope {scope} not allowed")
    sessions = [
        SessionData.model_validate_json(body)
        for body in store.database.all("session")
    ]
    if not any(
        session.scope == scope
        and any(entry.id == source_entry_id for entry in sessions)
        for session in sessions
    ):
        raise ValueError(f"session {source_entry_id} not allowed")
    for body in store.database.all("memory"):
        memory = Memory.model_validate_json(body)
        if (
            memory.active
            and memory.scope == scope
            and memory.text == text
        ):
            return memory
    memory = Memory(
        text=text,
        scope=scope,
        source_entry_id=source_entry_id,
    )
    store.database.put("memory", memory.id, memory.model_dump_json())
    return memory


def recall(
        store: MemoryStore, query: str, allowed_scopes: frozenset[str], limit: int = 5
) -> list[Memory]:
    if not allowed_scopes <= store.allowed_scopes:
        raise ValueError(f"scope {allowed_scopes} not allowed")
    if not query.strip():
        return []
    keyword = query.casefold().split()
    ranked:list[tuple[int,Memory]] = []
    seen:list[tuple[str,str]] = []
    result:list[Memory] = []
    for body in store.database.all("memory"):
        memory = Memory.model_validate_json(body)
        if not memory.active or memory.scope not in allowed_scopes:
            continue
        score = sum(word in memory.text for word in keyword)
        if score:
            ranked.append((score, memory))
    ranked.sort(key=lambda pair:(-pair[0], pair[1].id))
    for _,memory in ranked:
        key = (memory.scope, memory.text)
        if key not in seen:
            seen.append(key)
            result.append(memory)
        if len(result) >= limit:
            break
    return result


def forget(store: MemoryStore, memory_id: str, *, confirmed: bool) -> None:
    if not confirmed:
        raise PermissionError("explicit confirmation required")
    with store.database.transaction():
        body = store.database.get("memory", memory_id)
        if body is None:
            raise KeyError("unknown memory")
        memory = Memory.model_validate_json(body)
        if memory.scope not in store.allowed_scopes:
            raise PermissionError("memory belongs to another scope")
        memory.active = False
        store.database.put("memory", memory.id, memory.model_dump_json())
