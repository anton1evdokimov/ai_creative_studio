"""CER of OCR text on a generated ad vs product-label reference."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


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


def _prep_ocr_views(im: Image.Image) -> list[Image.Image]:
    rgb = im.convert("RGB")
    w, h = rgb.size
    short = min(w, h)
    if short < 1100:
        s = 1100 / max(short, 1)
        rgb = rgb.resize((max(1, int(w * s)), max(1, int(h * s))), Image.Resampling.LANCZOS)
    gray = ImageOps.grayscale(rgb)
    gray = ImageEnhance.Contrast(gray).enhance(2.2)
    gray = ImageEnhance.Sharpness(gray).enhance(1.8)
    views = [gray, ImageOps.autocontrast(gray), gray.filter(ImageFilter.SHARPEN)]
    import numpy as np

    mean = float(np.asarray(gray).mean())
    if mean < 100:
        views.append(ImageOps.invert(gray))
    return views


def ocr_image(path: str | Path, lang: str = "rus+eng") -> str:
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError("pip install pytesseract and apt/brew install tesseract") from exc
    im = Image.open(path).convert("RGB")
    langs = [lang]
    if "+" in (lang or ""):
        langs.extend(x.strip() for x in lang.split("+") if x.strip() and x.strip() != lang)
    configs = ("--oem 3 --psm 6", "--oem 3 --psm 11", "--oem 3 --psm 4", "--oem 3 --psm 7")
    best = ""
    for view in _prep_ocr_views(im):
        for lg in langs:
            for cfg in configs:
                try:
                    t = pytesseract.image_to_string(view, lang=lg, config=cfg) or ""
                except Exception:
                    continue
                if len(t.strip()) > len(best.strip()):
                    best = t
    return best


def score_cer(
    image_path: str | Path,
    product: dict | None,
    lang: str = "rus+eng",
) -> tuple[float | None, float | None, str]:
    """Returns (cer 0..1, cer_score 1-cer, ocr_text). None if no reference or OCR empty."""
    ref = reference_label(product)
    if not ref:
        return None, None, ""
    hyp = ocr_image(image_path, lang=lang)
    ocr_txt = " ".join(hyp.split())[:240]
    if not normalize_label(hyp):
        return None, None, ocr_txt
    cer = character_error_rate(hyp, ref)
    return round(cer, 4), round(1.0 - cer, 4), ocr_txt
