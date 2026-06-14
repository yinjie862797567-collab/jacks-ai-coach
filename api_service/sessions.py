import uuid
import time
from typing import Optional

MAX_HISTORY = 50
sessions: dict[str, list[dict]] = {}


def generate_id() -> str:
    return uuid.uuid4().hex[:12]


def get_or_create(session_id: Optional[str]) -> tuple[str, list[dict]]:
    if session_id and session_id in sessions:
        return session_id, sessions[session_id]
    sid = generate_id()
    sessions[sid] = []
    return sid, sessions[sid]


def add_message(session_id: str, role: str, content: str):
    if session_id not in sessions:
        sessions[session_id] = []
    sessions[session_id].append({
        "role": role,
        "content": content,
        "timestamp": time.time(),
    })
    if len(sessions[session_id]) > MAX_HISTORY:
        sessions[session_id] = sessions[session_id][-MAX_HISTORY:]


def clear(session_id: str) -> bool:
    if session_id in sessions:
        del sessions[session_id]
        return True
    return False
