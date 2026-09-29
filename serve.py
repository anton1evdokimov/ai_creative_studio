"""HTTP wrapper: upload UI + /generate. One GPU, one request at a time."""
from __future__ import annotations

import html
import json
import os
import threading
import uuid
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from agent.graph import build_graph
from agent.persist import save_graph_state
from models.media import ingest_image_bytes
from models.prompt_spec import parse_scene_prompt
from serve_jobs import attach_job_inputs, create_job, get_job, run_exclusive, set_jobs_root

INPUT_DIR = Path(os.environ.get("AICS_INPUT_DIR", "/data/input"))
OUTPUT_ROOT = Path(os.environ.get("AICS_OUTPUT_DIR", "/app/generated"))

app = FastAPI(title="AI Creative Studio")
_graph = None

_FORM = """<!DOCTYPE html>
<html lang="ru"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Creative Studio</title>
<style>
body{font-family:system-ui,sans-serif;background:#111;color:#eee;max-width:42rem;margin:2.5rem auto;padding:0 1.2rem}
h1{font-size:1.35rem;font-weight:600}
.card{background:#1c1c1c;border:1px solid #333;border-radius:12px;padding:1.4rem}
label{display:block;margin:.8rem 0 .35rem;color:#aaa;font-size:.9rem}
input[type=file],input[type=text],textarea{width:100%;box-sizing:border-box;background:#111;color:#eee;border:1px solid #444;border-radius:8px;padding:.55rem}
button{margin-top:1.1rem;background:#6c5ce7;border:0;color:#fff;padding:.7rem 1.2rem;border-radius:8px;font-size:1rem;cursor:pointer}
button.secondary{background:#333;margin-left:.5rem}
button:disabled{opacity:.5}
label.chk{display:flex;align-items:center;gap:.55rem;margin-top:1rem;color:#ccc;font-size:.95rem}
label.chk input{width:auto;margin:0}
.hint{color:#888;font-size:.85rem;margin-top:.8rem}
</style></head><body>
<h1>AI Creative Studio</h1>
<div class="card">
<form id="f" action="/generate" method="post" enctype="multipart/form-data">
<label>Фото продукта</label>
<input type="file" name="image" accept="image/*" required>
<label>Описание продукта (необязательно)</label>
<input type="text" name="description" placeholder="кефир, худи, сыворотка…">
<label>Промпт сцены (JSON или текст)</label>
<textarea name="scene" rows="8" placeholder='{"scene":"rustic wooden table, morning light","lighting":"soft window light from the left","mood":"warm"}'></textarea>
<label class="chk"><input type="checkbox" name="video" value="1"> Генерировать видео</label>
<button type="submit" id="go">Сгенерировать</button>
<button type="button" id="restart" class="secondary">Перезапустить генерацию</button>
<p class="hint">Сцена: JSON с полями scene, lighting, camera, style, mood — или одна строка. Запрос в очередь GPU. Перезапуск — те же фото, описание и сцена.</p>
</form>
</div>
<script>
let pollGen = 0;
const form = document.getElementById("f");
const go = document.getElementById("go");
async function startJob() {
  const my = ++pollGen;
  go.disabled = true;
  go.textContent = "В очереди…";
  const fd = new FormData(form);
  const res = await fetch("/jobs", {method: "POST", body: fd});
  const data = await res.json();
  if (!res.ok) { go.disabled = false; go.textContent = data.detail || "error"; return; }
  const id = data.job_id;
  const tick = async () => {
    if (my !== pollGen) return;
    const st = await (await fetch("/jobs/" + id)).json();
    if (my !== pollGen) return;
    go.textContent = st.status || "…";
    if (st.status === "done") { window.location = "/jobs/" + id + "?view=html"; return; }
    if (st.status === "awaiting_human") { window.location = "/jobs/" + id + "?view=html"; return; }
    if (st.status === "error") { go.disabled = false; go.textContent = st.error || "error"; return; }
    setTimeout(tick, 3000);
  };
  tick();
}
form.addEventListener("submit", function(ev){ ev.preventDefault(); startJob(); });
document.getElementById("restart").addEventListener("click", function(){ startJob(); });
</script>
</body></html>
"""


def _wants_html(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    return "text/html" in accept and "application/json" not in accept.split(",")[0]


def _file_url(path: str) -> str:
    return "/files?path=" + quote(path, safe="")


def _result_html(payload: dict) -> str:
    cards = []
    for s in payload.get("scores") or []:
        p = s.get("path") or ""
        src = html.escape(_file_url(p))
        name = html.escape(Path(p).name)
        cards.append(
            f'<figure><img src="{src}" alt="{name}">'
            f"<figcaption>{name} · score={s.get('score')} · "
            f"CLIP-T={s.get('clip_t')} · CLIP-I={s.get('clip_i')} · CER={s.get('cer')}</figcaption></figure>"
        )
    best = payload.get("best_image") or ""
    best_block = (
        f'<p>Лучший кадр: <a href="{html.escape(_file_url(best))}">{html.escape(Path(best).name)}</a></p>'
        if best
        else "<p>Кадры не получились.</p>"
    )
    vids = payload.get("generated_videos") or []
    vhtml = ""
    for v in vids:
        href = html.escape(_file_url(v))
        vhtml += f'<p>Видео: <a href="{href}">{html.escape(Path(v).name)}</a></p>'
    prompts = payload.get("prompts") or []
    phtml = ""
    if prompts:
        blob = html.escape(json.dumps(prompts, ensure_ascii=False, indent=2))
        phtml = f"<h2>Промпты (JSON)</h2><pre style=\"overflow:auto;background:#111;border:1px solid #333;padding:1rem;border-radius:8px;font-size:.8rem\">{blob}</pre>"
    jid = html.escape(str(payload.get("job") or ""))
    return f"""<!DOCTYPE html>
<html lang="ru"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Результат</title>
<style>
body{{font-family:system-ui,sans-serif;background:#111;color:#eee;max-width:52rem;margin:2rem auto;padding:0 1.2rem}}
a{{color:#a29bfe}}
img{{max-width:100%;border-radius:10px;border:1px solid #333}}
figure{{margin:1.2rem 0}}
figcaption{{color:#888;font-size:.85rem;margin-top:.4rem}}
button{{margin:.8rem .5rem 0 0;background:#6c5ce7;border:0;color:#fff;padding:.7rem 1.2rem;border-radius:8px;font-size:1rem;cursor:pointer}}
button.secondary{{background:#333}}
button:disabled{{opacity:.5}}
</style></head><body>
<p><a href="/generate">← новое фото</a></p>
<h1>Готово</h1>
<button type="button" id="restart">Перезапустить генерацию</button>
<a href="/generate"><button type="button" class="secondary">Другое фото</button></a>
{best_block}
{vhtml}
{phtml}
{''.join(cards) or '<p>Нет оценок.</p>'}
<script>
const restart = document.getElementById("restart");
restart.addEventListener("click", async () => {{
  restart.disabled = true;
  restart.textContent = "В очереди…";
  const res = await fetch("/jobs/{jid}/rerun", {{method: "POST"}});
  const data = await res.json();
  if (!res.ok) {{ restart.disabled = false; restart.textContent = data.detail || "error"; return; }}
  const id = data.job_id;
  const tick = async () => {{
    const st = await (await fetch("/jobs/" + id)).json();
    restart.textContent = st.status || "…";
    if (st.status === "done") {{ window.location = "/jobs/" + id + "?view=html"; return; }}
    if (st.status === "error") {{ restart.disabled = false; restart.textContent = st.error || "error"; return; }}
    setTimeout(tick, 3000);
  }};
  tick();
}});
</script>
</body></html>
"""


@app.on_event("startup")
def _startup() -> None:
    global _graph
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    set_jobs_root(OUTPUT_ROOT)
    _graph = build_graph(db_path=str(OUTPUT_ROOT / "langgraph.sqlite"))


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _FORM


@app.get("/generate", response_class=HTMLResponse)
def generate_page() -> str:
    return _FORM


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


def _payload(result: dict, job_id: str) -> dict:
    evals = result.get("evaluation_typed") or []
    prompts = []
    for gr in result.get("generation_results") or []:
        prompts.append(
            {
                "path": getattr(gr, "image_path", None),
                "prompt": getattr(gr, "prompt_json", None) or {},
            }
        )
    return {
        "job": job_id,
        "best_image": result.get("best_image"),
        "generated_images": result.get("generated_images") or [],
        "generated_videos": result.get("generated_videos") or [],
        "scene_spec": result.get("scene_spec") or {},
        "prompts": prompts,
        "scores": [
            {
                "path": e.image_path,
                "score": e.score,
                "clip_t": e.clip_metrics.clip_t if e.clip_metrics else None,
                "clip_i": e.clip_metrics.clip_i if e.clip_metrics else None,
                "aesthetic": e.clip_metrics.aesthetic if e.clip_metrics else None,
                "cer": e.cer,
                "cer_score": e.cer_score,
                "ocr_text": e.ocr_text,
            }
            for e in evals
        ],
    }


def _graph_config(jid: str) -> dict:
    return {"configurable": {"thread_id": jid}}


def _interrupt_payload(graph, config: dict):
    try:
        snap = graph.get_state(config)
    except Exception:
        return None
    nxt = tuple(getattr(snap, "next", None) or ())
    if not nxt:
        return None
    for task in getattr(snap, "tasks", None) or ():
        for item in getattr(task, "interrupts", None) or ():
            val = getattr(item, "value", item)
            if val is not None:
                return val
    return {"next": list(nxt)}


def _run_graph(
    image_path: str,
    description: str,
    scene: str = "",
    want_video: bool = False,
    jid: str = "cli",
) -> dict:
    if _graph is None:
        raise RuntimeError("Graph not ready")
    spec = parse_scene_prompt(scene)
    config = _graph_config(jid)
    inputs = {
        "product_image": image_path,
        "product_description": description,
        "scene_prompt": scene,
        "scene_spec": spec,
        "want_video": bool(want_video),
        "retry_count": 0,
    }
    try:
        result = _graph.invoke(inputs, config)
    except Exception:
        paused = _interrupt_payload(_graph, config)
        if paused is not None:
            return {"_paused": True, "interrupt": paused}
        raise
    try:
        save_graph_state(OUTPUT_ROOT, jid, result if isinstance(result, dict) else {})
    except Exception:
        pass
    paused = _interrupt_payload(_graph, config)
    if paused is not None:
        return {"_paused": True, "interrupt": paused}
    return result


def _gate_html(jid: str, job: dict) -> str:
    payload = job.get("interrupt") or {}
    if not isinstance(payload, dict):
        payload = {}
    cards = []
    for s in payload.get("images") or []:
        if not isinstance(s, dict):
            continue
        p = s.get("image") or s.get("path") or ""
        if not p:
            continue
        src = html.escape(_file_url(p))
        name = html.escape(Path(p).name)
        score = html.escape(str(s.get("score", "")))
        cards.append(
            f'<label class="pick"><input type="radio" name="best" value="{html.escape(p)}">'
            f'<img src="{src}" alt="{name}"><span>{name} · {score}</span></label>'
        )
    want = "checked" if payload.get("want_video") else ""
    return f"""<!DOCTYPE html>
<html lang="ru"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Выбор кадра</title>
<style>
body{{font-family:system-ui,sans-serif;background:#111;color:#eee;max-width:52rem;margin:2rem auto;padding:0 1.2rem}}
.pick{{display:block;margin:1rem 0;cursor:pointer}}
img{{max-width:100%;border-radius:10px;border:1px solid #333}}
button{{margin:.5rem .4rem 0 0;background:#6c5ce7;border:0;color:#fff;padding:.6rem 1rem;border-radius:8px;cursor:pointer}}
</style></head><body>
<h1>Human-in-the-loop</h1>
<p>Выбери кадр, видео, дальше / retry / стоп.</p>
<form id="g">
{''.join(cards) or '<p>Нет кадров.</p>'}
<p><label><input type="checkbox" id="vid" {want}> видео</label></p>
<button type="button" data-a="continue">Дальше</button>
<button type="button" data-a="retry">Retry промпт</button>
<button type="button" data-a="end">Стоп</button>
</form>
<script>
const jid = {json.dumps(jid)};
document.querySelectorAll("button[data-a]").forEach(btn => btn.onclick = async () => {{
  const best = (document.querySelector("input[name=best]:checked") || {{}}).value;
  const body = {{
    action: btn.dataset.a,
    want_video: document.getElementById("vid").checked,
  }};
  if (best) body.best_image = best;
  btn.disabled = true;
  await fetch("/jobs/" + jid + "/resume", {{method: "POST", headers: {{"Content-Type": "application/json"}}, body: JSON.stringify(body)}});
  const tick = async () => {{
    const st = await (await fetch("/jobs/" + jid)).json();
    if (st.status === "done") {{ window.location = "/jobs/" + jid + "?view=html"; return; }}
    if (st.status === "error") {{ alert(st.error || "error"); return; }}
    if (st.status === "awaiting_human") {{ window.location.reload(); return; }}
    setTimeout(tick, 3000);
  }};
  tick();
}});
</script>
</body></html>
"""


@app.post("/jobs")
async def create_pipeline_job(
    image: UploadFile = File(...),
    description: str = Form(""),
    scene: str = Form(""),
    video: str = Form(""),
):
    if _graph is None:
        raise HTTPException(503, "Graph not ready")
    dest = INPUT_DIR / uuid.uuid4().hex[:12]
    try:
        dest = ingest_image_bytes(await image.read(), image.filename or "product.png", dest)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    jid = create_job()
    desc = description
    scene_txt = scene
    want_video = str(video).lower() in {"1", "true", "on", "yes"}
    path = str(dest)
    attach_job_inputs(jid, path, desc, scene_txt, want_video)

    def work():
        raw = _run_graph(path, desc, scene_txt, want_video, jid)
        if isinstance(raw, dict) and raw.get("_paused"):
            return raw
        return _payload(raw, jid)

    threading.Thread(target=lambda: run_exclusive(jid, work), daemon=True).start()
    return JSONResponse({"job_id": jid, "status": "queued"}, status_code=202)


@app.get("/jobs/{jid}")
def job_status(jid: str, view: str = ""):
    job = get_job(jid)
    if job is None:
        raise HTTPException(404, "unknown job")
    if view == "html" and job.get("status") == "awaiting_human":
        return HTMLResponse(_gate_html(jid, job))
    if view == "html" and job.get("status") == "done" and job.get("result"):
        return HTMLResponse(_result_html(job["result"]))
    body = {k: job[k] for k in ("status", "error") if k in job}
    if job.get("interrupt"):
        body["interrupt"] = job["interrupt"]
    if job.get("result"):
        body["result"] = job["result"]
    return JSONResponse(body)


@app.post("/jobs/{jid}/rerun")
def rerun_job(jid: str):
    old = get_job(jid)
    if old is None or not old.get("image_path"):
        raise HTTPException(404, "unknown job")
    path = str(old["image_path"])
    desc = str(old.get("description") or "")
    scene_txt = str(old.get("scene") or "")
    want_video = bool(old.get("want_video"))
    if not Path(path).is_file():
        raise HTTPException(400, "source image no longer on disk")
    new_id = create_job()
    attach_job_inputs(new_id, path, desc, scene_txt, want_video)

    def work():
        raw = _run_graph(path, desc, scene_txt, want_video, new_id)
        if isinstance(raw, dict) and raw.get("_paused"):
            return raw
        return _payload(raw, new_id)

    threading.Thread(target=lambda: run_exclusive(new_id, work), daemon=True).start()
    return JSONResponse({"job_id": new_id, "status": "queued"}, status_code=202)


@app.post("/jobs/{jid}/resume")
async def resume_job(jid: str, request: Request):
    if _graph is None:
        raise HTTPException(503, "Graph not ready")
    job = get_job(jid)
    if job is None:
        raise HTTPException(404, "unknown job")
    if job.get("status") != "awaiting_human":
        raise HTTPException(409, f"job is {job.get('status')}, not awaiting_human")
    body = await request.json()
    if not isinstance(body, dict):
        body = {}

    def work():
        from langgraph.types import Command

        config = _graph_config(jid)
        try:
            result = _graph.invoke(Command(resume=body), config)
        except Exception:
            paused = _interrupt_payload(_graph, config)
            if paused is not None:
                return {"_paused": True, "interrupt": paused}
            raise
        try:
            save_graph_state(OUTPUT_ROOT, jid, result if isinstance(result, dict) else {})
        except Exception:
            pass
        paused = _interrupt_payload(_graph, config)
        if paused is not None:
            return {"_paused": True, "interrupt": paused}
        return _payload(result, jid)

    threading.Thread(target=lambda: run_exclusive(jid, work), daemon=True).start()
    return JSONResponse({"job_id": jid, "status": "queued"}, status_code=202)


@app.post("/generate")
async def generate(
    request: Request,
    image: UploadFile = File(...),
    description: str = Form(""),
    scene: str = Form(""),
    video: str = Form(""),
):
    """Enqueue on GPU (same as POST /jobs). Sync wait is not used — one GPU lock is inside the worker."""
    return await create_pipeline_job(image=image, description=description, scene=scene, video=video)


@app.get("/files")
def get_file(path: str):
    p = Path(path).resolve()
    allowed = (OUTPUT_ROOT.resolve(), Path("/app/generated").resolve(), INPUT_DIR.resolve())
    if not p.is_file() or not any(str(p).startswith(str(a)) for a in allowed):
        raise HTTPException(404, "file not found")
    return FileResponse(p)
