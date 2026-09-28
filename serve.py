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

INPUT_DIR = Path(os.environ.get("AICS_INPUT_DIR", "/data/input"))
OUTPUT_ROOT = Path(os.environ.get("AICS_OUTPUT_DIR", "/app/generated"))

app = FastAPI(title="AI Creative Studio")
_graph = None
_lock = threading.Lock()

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
<button type="submit">Сгенерировать</button>
<p class="hint">Один запрос занимает GPU на 5–20 минут. Не закрывайте вкладку.</p>
</form>
</div>
<script>
document.getElementById("f").addEventListener("submit", function(){
  this.querySelector("button").disabled = true;
  this.querySelector("button").textContent = "Генерация…";
});
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
            f"CLIP-T={s.get('clip_t')} · CLIP-I={s.get('clip_i')}</figcaption></figure>"
        )
    best = payload.get("best_image") or ""
    best_block = (
        f'<p>Лучший кадр: <a href="{html.escape(_file_url(best))}">{html.escape(Path(best).name)}</a></p>'
        if best
        else "<p>Кадры не получились.</p>"
    )
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
</style></head><body>
<p><a href="/">← новое фото</a></p>
<h1>Готово</h1>
{best_block}
{''.join(cards) or '<p>Нет оценок.</p>'}
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


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.post("/generate")
async def generate(
    request: Request,
    image: UploadFile = File(...),
    description: str = Form(""),
):
    if _graph is None:
        raise HTTPException(503, "Graph not ready")

    job = uuid.uuid4().hex[:12]
    dest = INPUT_DIR / job
    try:
        dest = ingest_image_bytes(await image.read(), image.filename or "product.png", dest)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    with _lock:
        result = _graph.invoke(
            {
                "product_image": str(dest),
                "product_description": description,
                "retry_count": 0,
            }
        )

    evals = result.get("evaluation_typed") or []
    payload = {
        "job": job,
        "best_image": result.get("best_image"),
        "generated_images": result.get("generated_images") or [],
        "scores": [
            {
                "path": e.image_path,
                "score": e.score,
                "clip_t": e.clip_metrics.clip_t if e.clip_metrics else None,
                "clip_i": e.clip_metrics.clip_i if e.clip_metrics else None,
                "aesthetic": e.clip_metrics.aesthetic if e.clip_metrics else None,
            }
            for e in evals
        ],
    }
    if _wants_html(request):
        return HTMLResponse(_result_html(payload))
    return JSONResponse(payload)


@app.get("/files")
def get_file(path: str):
    p = Path(path).resolve()
    allowed = (OUTPUT_ROOT.resolve(), Path("/app/generated").resolve(), INPUT_DIR.resolve())
    if not p.is_file() or not any(str(p).startswith(str(a)) for a in allowed):
        raise HTTPException(404, "file not found")
    return FileResponse(p)
