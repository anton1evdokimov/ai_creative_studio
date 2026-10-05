"""User scene lock + JSON generation prompt (flatten to text for SDXL)."""
from __future__ import annotations

import json
from typing import Any

from schemas.creative import CreativeConcept

SCENE_KEYS = (
    "scene",
    "lighting",
    "style",
    "mood",
    "camera",
    "camera_angle",
    "composition",
    "negative",
    "negative_prompt",
    "color_palette",
    "product",
)

_FLAT_ORDER = (
    "product",
    "scene",
    "lighting",
    "camera",
    "style",
    "mood",
    "composition",
)


def parse_scene_prompt(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {"scene": text}
        if isinstance(data, dict):
            out: dict[str, Any] = {}
            for k, v in data.items():
                key = str(k).strip()
                if not key or v in (None, "", [], {}):
                    continue
                out[key] = v
            if "camera" not in out and out.get("camera_angle"):
                out["camera"] = out["camera_angle"]
            return out
    return {"scene": text}


def flatten_prompt_json(spec: dict[str, Any] | None) -> str:
    if not spec:
        return ""
    parts: list[str] = []
    for k in _FLAT_ORDER:
        v = spec.get(k) or (spec.get("camera_angle") if k == "camera" else None)
        if v in (None, "", [], {}):
            continue
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v if x)
        s = str(v).strip().rstrip(",")
        if s:
            parts.append(s)
    tags = spec.get("style_boost_tags") or spec.get("tags") or []
    if isinstance(tags, list):
        extra = ", ".join(str(t).strip() for t in tags if str(t).strip())
        if extra:
            parts.append(extra)
    return ", ".join(parts)


def user_kandinsky_json(
    scene_raw: str = "",
    description: str = "",
    spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build Kandinsky prompt JSON from the user's scene/description only (no LLM)."""
    data = dict(spec) if spec else parse_scene_prompt(scene_raw)
    desc = (description or "").strip()
    if desc and not data.get("product"):
        data["product"] = desc
    if not data:
        text = (scene_raw or desc).strip() or "scene"
        data = {"scene": text}
    return data


def user_kandinsky_prompt(
    scene_raw: str = "",
    description: str = "",
    spec: dict[str, Any] | None = None,
    as_json: bool = True,
) -> tuple[dict[str, Any], str]:
    data = user_kandinsky_json(scene_raw, description, spec)
    if as_json:
        return data, json.dumps(data, ensure_ascii=False)
    return data, flatten_prompt_json(data) or str(data.get("scene") or "")


def apply_scene_to_concept(concept: CreativeConcept, spec: dict[str, Any] | None) -> CreativeConcept:
    if not spec:
        return concept
    upd: dict[str, Any] = {}
    if spec.get("scene"):
        upd["scene"] = str(spec["scene"])
    if spec.get("lighting"):
        upd["lighting"] = str(spec["lighting"])
    if spec.get("style"):
        upd["style"] = str(spec["style"])
    if spec.get("mood"):
        upd["mood"] = str(spec["mood"])
    cam = spec.get("camera") or spec.get("camera_angle")
    if cam:
        upd["camera_angle"] = str(cam)
    if spec.get("composition"):
        upd["composition"] = str(spec["composition"])
    pal = spec.get("color_palette")
    if isinstance(pal, list) and pal:
        upd["color_palette"] = [str(x) for x in pal]
    tags = spec.get("tags")
    if isinstance(tags, list) and tags:
        upd["tags"] = [str(x) for x in tags]
    return concept.model_copy(update=upd) if upd else concept


def negative_from_spec(spec: dict[str, Any] | None) -> str:
    if not spec:
        return ""
    return str(spec.get("negative_prompt") or spec.get("negative") or "").strip()
