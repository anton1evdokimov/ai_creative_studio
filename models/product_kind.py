"""Detect apparel vs packaged goods so fallbacks/prompts are not cosmetics-only."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from schemas.creative import CreativeConcept


APPAREL_TOKENS = (
    "hoodie", "hoodies", "sweatshirt", "sweater", "jumper", "pullover",
    "t-shirt", "tshirt", "tee", "shirt", "polo",
    "jacket", "coat", "parka", "bomber", "blazer",
    "jeans", "pants", "trousers", "shorts", "skirt", "dress",
    "sneakers", "shoes", "boots", "trainer",
    "apparel", "clothing", "garment", "fashion", "wear",
    "fleece", "knit",
)

COSMETIC_TOKENS = (
    "serum", "cream", "skincare", "lotion", "cosmetic", "beauty",
    "bottle", "dropper", "fragrance", "perfume", "lipstick",
)


def _as_dict(analysis: Any) -> dict:
    if analysis is None:
        return {}
    if isinstance(analysis, dict):
        return analysis
    try:
        return analysis.model_dump()
    except Exception:
        return {}


def _blob(*parts: Any) -> str:
    bits = []
    for p in parts:
        if not p:
            continue
        if isinstance(p, dict):
            bits.append(" ".join(str(v) for v in p.values() if v))
        else:
            bits.append(str(p))
    return " ".join(bits).lower()


def looks_like_apparel(analysis: Any = None, description: str = "", image_path: str | Path = "") -> bool:
    name = Path(image_path).stem if image_path else ""
    data = _as_dict(analysis)
    text = _blob(
        description,
        name,
        data.get("category"),
        data.get("product_type"),
        data.get("product_name"),
        data.get("visual_caption"),
        " ".join(data.get("key_visual_features") or []),
    )
    if any(t in text for t in APPAREL_TOKENS):
        return True
    if any(t in text for t in COSMETIC_TOKENS):
        return False
    return False


def filename_hint(image_path: str | Path = "") -> str:
    stem = Path(image_path).stem if image_path else ""
    cleaned = stem.replace("_", " ").replace("-", " ").strip()
    low = cleaned.lower()
    if not cleaned:
        return ""
    if any(t in low for t in APPAREL_TOKENS) or any(t in low for t in COSMETIC_TOKENS):
        return cleaned
    if cleaned.isalpha() and 3 <= len(cleaned) <= 24:
        return cleaned
    return ""


def product_noun(
    analysis: Any = None,
    description: str = "",
    image_path: str | Path = "",
) -> str:
    data = _as_dict(analysis)
    for key in ("product_name", "product_type", "visual_caption"):
        val = str(data.get(key) or "").strip()
        if val:
            return val.split(".")[0][:120]
    desc = " ".join((description or "").split())
    if desc and desc.lower() not in {"retail product", "retail consumer product", "."}:
        return desc[:120]
    hint = filename_hint(image_path)
    if hint:
        return hint
    return "product"


def ensure_product_lead(prompt: str, noun: str) -> str:
    text = " ".join((prompt or "").split())
    lead = (noun or "").strip()
    if not lead:
        return text
    head = text[:160].lower()
    if lead.lower() in head:
        return text
    return f"{lead}, {text}".strip(" ,")


def fallback_concepts(apparel: bool) -> list[CreativeConcept]:
    if apparel:
        return [
            CreativeConcept(
                name="Ghost Mannequin Studio",
                scene="apparel on invisible ghost mannequin, clean white cyclorama, garment hanging with natural drape",
                lighting="soft large octabox 45deg front-left, gentle rim to separate fabric from backdrop",
                style="high-end ecommerce fashion photography, lookbook catalog",
                mood="clean, premium, contemporary",
                camera_angle="eye-level, three-quarter view of the garment",
                color_palette=["#F4F4F4", "#111111", "#C8C8C8"],
                composition="garment fills 60% of frame, centered, space around silhouette",
                tags=["ghost mannequin", "lookbook", "studio", "fashion ecommerce", "fabric drape"],
            ),
            CreativeConcept(
                name="Street Lookbook",
                scene="garment as hero on a simple urban wall, golden hour, no distracting props, fabric texture visible",
                lighting="warm late-afternoon sunlight from camera-right, white bounce fill",
                style="editorial street fashion lookbook, commercial campaign",
                mood="casual, aspirational, urban",
                camera_angle="slightly low three-quarter, medium shot",
                color_palette=["#D4A574", "#4A4A4A", "#E8E0D4"],
                composition="rule of thirds, garment as the only subject",
                tags=["lookbook", "street", "fashion", "editorial", "fabric"],
            ),
        ]
    return [
        CreativeConcept(
            name="Minimalist Studio",
            scene="white seamless studio backdrop, single pale travertine pedestal",
            lighting="soft diffused studio key light 45deg top-left + subtle cool rim light behind",
            style="high-end commercial product photography, Peter Lippmann style",
            mood="clean, serene, premium, calm",
            camera_angle="eye-level straight on, medium close-up",
            color_palette=["#FFFFFF", "#F5F5F5", "#E8E0D4", "#C9A227"],
            composition="centered symmetry, generous negative space all around",
            tags=["studio shot", "clean", "minimal", "pedestal", "magazine", "editorial"],
        ),
        CreativeConcept(
            name="Golden Botanical Hour",
            scene="product on natural travertine slab, soft green eucalyptus leaves bokeh around edges, thin window blind shadows",
            lighting="golden hour warm sunlight 35deg backlight, white foamcore fill",
            style="organic lifestyle editorial, natural light photography",
            mood="warm, fresh, natural, aspirational, calm",
            camera_angle="45-degree, medium shot",
            color_palette=["#D4A574", "#8B9A6B", "#F7E7CE", "#E6B325"],
            composition="rule of thirds, product upper-left, negative space for headline",
            tags=["lifestyle", "natural", "golden hour", "botanical", "editorial"],
        ),
    ]


def category_negative(apparel: bool) -> str:
    shared = (
        "jpeg artifacts, watermarks, captions, cartoon, anime, illustration, "
        "lowres, blurry, extra limbs, bad anatomy, deformed"
    )
    if apparel:
        return (
            "extra hoods, extra sleeves, melted fabric, fused pockets, warped collar, "
            "asymmetric shoulders, floating garment, mannequin seams, extra zipper, "
            "deformed print, extra legs, " + shared
        )
    return (
        "melted glass, broken bottle, extra dropper, warped typography, "
        "wrong perspective glass, duplicate bottles, " + shared
    )
