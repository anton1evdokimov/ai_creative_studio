"""JSON snapshots of pipeline / HTTP jobs (survive uvicorn restart)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except Exception:
            return str(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(x) for x in obj]
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def jobs_dir(root: str | Path) -> Path:
    p = Path(root) / "jobs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_job(root: str | Path, jid: str, job: dict[str, Any]) -> None:
    path = jobs_dir(root) / f"{jid}.json"
    dump = {k: jsonable(v) for k, v in job.items() if k != "traceback"}
    if job.get("traceback"):
        dump["traceback"] = str(job["traceback"])[-8000:]
    path.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")


def load_job(root: str | Path, jid: str) -> dict[str, Any] | None:
    path = jobs_dir(root) / f"{jid}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_graph_state(root: str | Path, jid: str, state: dict[str, Any]) -> None:
    path = jobs_dir(root) / f"{jid}_graph.json"
    path.write_text(json.dumps(jsonable(state), ensure_ascii=False, indent=2), encoding="utf-8")
