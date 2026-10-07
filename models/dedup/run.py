"""S3 listing → SHA-256 → pHash → DINOv2 → FAISS → dedup report."""
from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

from PIL import Image

from models.storage.s3store import get_bytes, list_image_keys


def _settings() -> dict:
    from models.diffusion.config import _load_root_config

    raw = _load_root_config().get("dedup") or {}
    if not isinstance(raw, dict):
        raw = {}
    return {
        "batch_size": max(1, int(raw.get("batch_size") or 8)),
        "phash_max_distance": int(raw.get("phash_max_distance") or 8),
        "dino_min_cosine": float(raw.get("dino_min_cosine") or 0.94),
        "dino_model": str(raw.get("dino_model") or "facebook/dinov2-small"),
    }


def _batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _phash(data: bytes):
    import imagehash

    im = Image.open(BytesIO(data))
    im.load()
    return imagehash.phash(im.convert("RGB"))


def deduplicate_prefix(prefix: str) -> dict:
    cfg = _settings()
    keys = list_image_keys(prefix)
    print(f"🧹 dedup list {len(keys)} object(s) under {prefix}", flush=True)
    dropped: list[dict] = []
    sha_kept: list[dict] = []
    seen_sha: dict[str, str] = {}

    for batch in _batches(keys, cfg["batch_size"]):
        for key in batch:
            data = get_bytes(key)
            digest = hashlib.sha256(data).hexdigest()
            prev = seen_sha.get(digest)
            if prev:
                dropped.append({"key": key, "stage": "sha256", "duplicate_of": prev, "sha256": digest})
                print(f"   sha256 drop {key} == {prev}", flush=True)
                continue
            seen_sha[digest] = key
            sha_kept.append({"key": key, "sha256": digest, "phash": str(_phash(data))})

    print(f"🧹 dedup sha256 kept {len(sha_kept)} / {len(keys)}", flush=True)

    import imagehash

    phash_kept: list[dict] = []
    kept_hashes = []
    limit = cfg["phash_max_distance"]
    for row in sha_kept:
        digest = imagehash.hex_to_hash(row["phash"])
        twin = None
        dist = None
        for prev_row, prev_hash in kept_hashes:
            d = int(digest - prev_hash)
            if d <= limit:
                twin = prev_row["key"]
                dist = d
                break
        if twin:
            dropped.append({
                "key": row["key"],
                "stage": "phash",
                "duplicate_of": twin,
                "distance": dist,
                "sha256": row["sha256"],
            })
            print(f"   phash drop {row['key']} ~ {twin} d={dist}", flush=True)
            continue
        phash_kept.append(row)
        kept_hashes.append((row, digest))

    print(f"🧹 dedup phash kept {len(phash_kept)}", flush=True)

    faiss_kept = _faiss_filter(phash_kept, dropped, cfg)
    report = {
        "prefix": prefix,
        "input_count": len(keys),
        "kept_count": len(faiss_kept),
        "kept": faiss_kept,
        "dropped": dropped,
        "thresholds": {
            "phash_max_distance": limit,
            "dino_min_cosine": cfg["dino_min_cosine"],
            "dino_model": cfg["dino_model"],
        },
    }
    print(f"🧹 dedup report kept {report['kept_count']} dropped {len(dropped)}", flush=True)
    return report


def _faiss_filter(rows: list[dict], dropped: list[dict], cfg: dict) -> list[dict]:
    if not rows:
        return []
    if len(rows) == 1:
        rows[0]["stage"] = "kept"
        return rows

    import numpy as np

    from models.ranking.dino_metrics import create_dino_metrics, unload_dino_metrics

    dino = create_dino_metrics(cfg["dino_model"])
    vectors = []
    try:
        for batch in _batches(rows, cfg["batch_size"]):
            for row in batch:
                im = Image.open(BytesIO(get_bytes(row["key"]))).convert("RGB")
                vec = dino.embed(im).detach().float().cpu().numpy().reshape(-1)
                vectors.append(vec)
            print(f"   dinov2 embedded {len(vectors)}/{len(rows)}", flush=True)
    finally:
        unload_dino_metrics()

    import faiss

    mat = np.stack(vectors).astype("float32")
    faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(mat.shape[1])
    kept_pos: list[int] = []
    thr = float(cfg["dino_min_cosine"])
    for i in range(mat.shape[0]):
        vec = mat[i : i + 1]
        if index.ntotal == 0:
            index.add(vec)
            kept_pos.append(i)
            continue
        dist, idx = index.search(vec, 1)
        cos = float(dist[0, 0])
        if cos >= thr:
            twin = rows[kept_pos[int(idx[0, 0])]]["key"]
            dropped.append({
                "key": rows[i]["key"],
                "stage": "faiss",
                "duplicate_of": twin,
                "cosine": round(cos, 4),
                "sha256": rows[i]["sha256"],
            })
            print(f"   faiss drop {rows[i]['key']} ~ {twin} cos={cos:.3f}", flush=True)
            continue
        index.add(vec)
        kept_pos.append(i)

    kept = []
    for i in kept_pos:
        row = dict(rows[i])
        row["stage"] = "kept"
        kept.append(row)
    print(f"🧹 dedup faiss kept {len(kept)}", flush=True)
    return kept


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
