"""CER of OCR text on a generated ad vs product-label reference."""
from __future__ import annotations

from pathlib import Path

from PIL import Image


def normalize_label(text: str) -> str:
    raw = (text or "").replace("\\n", "\n")
    return " ".join(raw.split()).lower()


def reference_label(product: dict | None) -> str:
    pa = product or {}
    ocr = str(pa.get("extracted_text_ocr") or "").strip()
    if ocr and "no legible" not in ocr.lower() and len(normalize_label(ocr)) >= 2:
        return ocr
    name = str(pa.get("product_name") or "").strip()
    return name if len(normalize_label(name)) >= 2 else ""


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins, delete, sub = prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def character_error_rate(hyp: str, ref: str) -> float:
    r = normalize_label(ref)
    h = normalize_label(hyp)
    if not r:
        return 0.0
    return min(1.0, levenshtein(h, r) / max(len(r), 1))


def ocr_image(path: str | Path, lang: str = "rus+eng") -> str:
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError("pip install pytesseract and apt/brew install tesseract") from exc
    im = Image.open(path).convert("RGB")
    w, h = im.size
    short = min(w, h)
    if short < 720:
        s = 720 / max(short, 1)
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.Resampling.LANCZOS)
    return pytesseract.image_to_string(im, lang=lang) or ""


def score_cer(
    image_path: str | Path,
    product: dict | None,
    lang: str = "rus+eng",
) -> tuple[float | None, float | None, str]:
    """Returns (cer 0..1, cer_score 1-cer, ocr_text). None if no reference."""
    ref = reference_label(product)
    if not ref:
        return None, None, ""
    hyp = ocr_image(image_path, lang=lang)
    cer = character_error_rate(hyp, ref)
    return round(cer, 4), round(1.0 - cer, 4), " ".join(hyp.split())[:240]
