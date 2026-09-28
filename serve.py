"""HTTP wrapper: upload UI + /generate. One GPU, one request at a time."""
from __future__ import annotations

import html
import os
import threading
import uuid
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from agent.graph import build_graph
from models.media import ingest_image_bytes
from serve_jobs import attach_job_inputs, create_job, get_job, run_exclusive

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
.hint{color:#888;font-size:.85rem;margin-top:.8rem}
</style></head><body>
<h1>AI Creative Studio</h1>
<div class="card">
<form id="f" action="/generate" method="post" enctype="multipart/form-data">
<label>Фото продукта</label>
<input type="file" name="image" accept="image/*" required>
<label>Описание (необязательно)</label>
<input type="text" name="description" placeholder="кефир, худи, сыворотка…">
<button type="submit" id="go">Сгенерировать</button>
<button type="button" id="restart" class="secondary">Перезапустить генерацию</button>
<p class="hint">Запрос уходит в очередь на GPU. Страница сама обновится, когда job станет done (5–20 мин). Перезапуск ставит новый job с теми же фото и описанием.</p>
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
    _graph = build_graph()


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
    return {
        "job": job_id,
        "best_image": result.get("best_image"),
        "generated_images": result.get("generated_images") or [],
        "generated_videos": result.get("generated_videos") or [],
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


def _run_graph(image_path: str, description: str) -> dict:
    if _graph is None:
        raise RuntimeError("Graph not ready")
    result = _graph.invoke(
        {
            "product_image": image_path,
            "product_description": description,
            "retry_count": 0,
        }
    )
    return result


@app.post("/jobs")
async def create_pipeline_job(
    image: UploadFile = File(...),
    description: str = Form(""),
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
    path = str(dest)
    attach_job_inputs(jid, path, desc)

    def work():
        return _payload(_run_graph(path, desc), jid)

    threading.Thread(target=lambda: run_exclusive(jid, work), daemon=True).start()
    return JSONResponse({"job_id": jid, "status": "queued"}, status_code=202)


@app.get("/jobs/{jid}")
def job_status(jid: str, view: str = ""):
    job = get_job(jid)
    if job is None:
        raise HTTPException(404, "unknown job")
    if view == "html" and job.get("status") == "done" and job.get("result"):
        return HTMLResponse(_result_html(job["result"]))
    body = {k: job[k] for k in ("status", "error") if k in job}
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
    if not Path(path).is_file():
        raise HTTPException(400, "source image no longer on disk")
    new_id = create_job()
    attach_job_inputs(new_id, path, desc)

    def work():
        return _payload(_run_graph(path, desc), new_id)

    threading.Thread(target=lambda: run_exclusive(new_id, work), daemon=True).start()
    return JSONResponse({"job_id": new_id, "status": "queued"}, status_code=202)


@app.post("/generate")
async def generate(
    request: Request,
    image: UploadFile = File(...),
    description: str = Form(""),
):
    """Enqueue on GPU (same as POST /jobs). Sync wait is not used — one GPU lock is inside the worker."""
    return await create_pipeline_job(image=image, description=description)


@app.get("/files")
def get_file(path: str):
    p = Path(path).resolve()
    allowed = (OUTPUT_ROOT.resolve(), Path("/app/generated").resolve(), INPUT_DIR.resolve())
    if not p.is_file() or not any(str(p).startswith(str(a)) for a in allowed):
        raise HTTPException(404, "file not found")
    return FileResponse(p)
