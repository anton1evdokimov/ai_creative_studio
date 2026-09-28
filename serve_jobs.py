"""One-GPU job queue for serve.py."""
from __future__ import annotations

import threading
import traceback
import uuid
from typing import Any, Callable

_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}


def create_job() -> str:
    jid = uuid.uuid4().hex[:12]
    _jobs[jid] = {"status": "queued", "error": None, "result": None}
    return jid


def get_job(jid: str) -> dict[str, Any] | None:
    return _jobs.get(jid)


def attach_job_inputs(jid: str, image_path: str, description: str) -> None:
    job = _jobs.get(jid)
    if job is None:
        return
    job["image_path"] = image_path
    job["description"] = description


def run_exclusive(jid: str, fn: Callable[[], dict]) -> None:
    job = _jobs.get(jid)
    if job is None:
        return
    with _lock:
        job["status"] = "running"
        try:
            job["result"] = fn()
            job["status"] = "done"
        except Exception as exc:
            job["status"] = "error"
            job["error"] = f"{type(exc).__name__}: {exc}"
            job["traceback"] = traceback.format_exc()
