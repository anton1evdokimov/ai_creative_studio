import json
import re
from typing import Any

from schemas.creative import CreativeConcepts, ConceptScore


def _extract_json_block(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first : last + 1]
    first = text.find("[")
    last = text.rfind("]")
    if first != -1 and last != -1 and last > first:
        return text[first : last + 1]
    return text


def _safe_json_loads(text: str, fallback: Any) -> Any:
    try:
        return json.loads(_extract_json_block(text))
    except Exception:
        return fallback


def parse_concepts(response: str) -> CreativeConcepts:
    data = _safe_json_loads(response, {"concepts": []})
    if isinstance(data, list):
        data = {"concepts": data}
    if "concepts" not in data or not isinstance(data["concepts"], list):
        data["concepts"] = []
    cleaned = []
    for c in data["concepts"]:
        if not isinstance(c, dict):
            continue
        c.setdefault("mood", "")
        c.setdefault("camera_angle", "")
        c.setdefault("color_palette", [])
        c.setdefault("composition", "")
        c.setdefault("tags", [])
        c.setdefault("scene", "")
        c.setdefault("lighting", "")
        c.setdefault("style", "")
        c.setdefault("name", c.get("name", "Concept")[:60])
        cleaned.append(c)
    data["concepts"] = cleaned
    return CreativeConcepts(**data)


def parse_concept_score(response: str) -> ConceptScore:
    data = _safe_json_loads(
        response,
        {
            "creativity": 5.0,
            "relevance": 5.0,
            "realizability": 5.0,
            "prompt_quality": 5.0,
            "avg": 5.0,
            "feedback": "",
        },
    )

    def _f(v, default=5.0):
        try:
            return max(1.0, min(10.0, float(v)))
        except Exception:
            return default

    creativity = _f(data.get("creativity", 5.0))
    relevance = _f(data.get("relevance", 5.0))
    realizability = _f(data.get("realizability", 5.0))
    prompt_quality = _f(data.get("prompt_quality", 5.0))
    avg = _f(
        data.get(
            "avg",
            (creativity + relevance + realizability + prompt_quality) / 4.0,
        )
    )
    feedback = str(data.get("feedback", ""))[:400]

    return ConceptScore(
        creativity=creativity,
        relevance=relevance,
        realizability=realizability,
        prompt_quality=prompt_quality,
        avg=avg,
        feedback=feedback,
    )