"""One-GPU job queue for serve.py. Jobs also written under generated/jobs/."""
from __future__ import annotations

import os
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable

from agent.persist import load_job, save_job

_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}
_JOBS_ROOT = Path(os.environ.get("AICS_OUTPUT_DIR", "generated"))


def set_jobs_root(path: str | Path) -> None:
    global _JOBS_ROOT
    _JOBS_ROOT = Path(path)


def _persist(jid: str) -> None:
    job = _jobs.get(jid)
    if job is None:
        return
    try:
        save_job(_JOBS_ROOT, jid, job)
    except Exception:
        pass


def create_job() -> str:
    jid = uuid.uuid4().hex[:12]
    _jobs[jid] = {"status": "queued", "error": None, "result": None}
    _persist(jid)
    return jid


def get_job(jid: str) -> dict[str, Any] | None:
    job = _jobs.get(jid)
    if job is not None:
        return job
    disk = load_job(_JOBS_ROOT, jid)
    if disk is not None:
        _jobs[jid] = disk
    return _jobs.get(jid)


def attach_job_inputs(
    jid: str,
    image_path: str,
    description: str,
    scene: str = "",
    want_video: bool = False,
    want_t2i: bool = False,
    want_direct: bool = False,
) -> None:
    job = get_job(jid)
    if job is None:
        return
    job["image_path"] = image_path
    job["description"] = description
    job["scene"] = scene
    job["want_video"] = want_video
    job["want_t2i"] = want_t2i
    job["want_direct"] = want_direct
    _persist(jid)


def run_exclusive(jid: str, fn: Callable[[], dict | None]) -> None:
    job = get_job(jid)
    if job is None:
        return
    with _lock:
        job["status"] = "running"
        _persist(jid)
        try:
            result = fn()
            if isinstance(result, dict) and result.get("_paused"):
                job["status"] = "awaiting_human"
                job["interrupt"] = result.get("interrupt")
                job["error"] = None
                _persist(jid)
                return
            job["result"] = result
            job["status"] = "done"
            job["interrupt"] = None
        except Exception as exc:
            job["status"] = "error"
            job["error"] = f"{type(exc).__name__}: {exc}"
            job["traceback"] = traceback.format_exc()
        _persist(jid)
