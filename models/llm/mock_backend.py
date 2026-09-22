import json
import random
import re
import time

from .base import LLMBackend


def _concepts_response(num_concepts: int) -> str:
    palette_presets = [
        ["#FFFFFF", "#F5F5F5", "#E8E0D4", "#C9B99E"],
        ["#2C3E50", "#34495E", "#95A5A6", "#ECF0F1"],
        ["#D4A574", "#8B9A6B", "#F7E7CE", "#784E2B"],
        ["#1A1A2E", "#16213E", "#E94560", "#0F3460"],
        ["#FAF3E0", "#EABF9F", "#B68973", "#1E212D"],
    ]
    presets = [
        {
            "name": "Minimalist Studio",
            "scene": "white seamless studio backdrop with a single beige travertine pedestal",
            "lighting": "soft diffused studio key light from left 45 degrees + subtle cool rim light behind",
            "style": "high-end commercial product photography, Peter Lippmann style, Hasselblad medium format",
            "mood": "clean, serene, premium, understated luxury",
            "camera_angle": "eye-level straight on, tight close-up framing",
            "composition": "centered symmetry, generous negative space top and right",
            "tags": ["studio shot", "clean", "minimal", "pedestal", "magazine quality"],
        },
        {
            "name": "Golden Botanical Hour",
            "scene": "product resting on a rough natural travertine slab with soft out-of-focus green monstera leaves",
            "lighting": "golden hour warm 3200K sunlight filtering through foliage, dappled leaf shadows",
            "style": "organic lifestyle editorial, natural light photography, Stephane Coutelle style",
            "mood": "warm, fresh, natural, aspirational wellbeing",
            "camera_angle": "45-degree flat lay overhead view",
            "composition": "rule of thirds, product placed at upper-left intersection",
            "tags": ["lifestyle", "natural light", "golden hour", "botanical", "outdoor"],
        },
        {
            "name": "Deep Blue Night Luxury",
            "scene": "bottle on glossy black marble, deep indigo velvet drape background, tiny bokeh specks",
            "lighting": "dramatic moody lighting, single cool spotlight from above + soft fill, rich blue gels",
            "style": "night luxury editorial, Quentin Jones style, cinematic noir",
            "mood": "dramatic, sensual, mysterious, exclusive",
            "camera_angle": "low angle hero shot looking up slightly",
            "composition": "strong diagonal, product bottom third, reflection on marble",
            "tags": ["noir", "luxury night", "glamour", "reflection", "marble"],
        },
        {
            "name": "Morning Vanity Ritual",
            "scene": "warm oak vanity countertop next to coffee mug, linen towel, gold perfume atomiser",
            "lighting": "soft overcast north-facing window light, no harsh shadows",
            "style": "lifestyle candid flat lay, authentic influencer aesthetic",
            "mood": "cozy, authentic, intimate, everyday self-care",
            "camera_angle": "35-degree flat lay slightly elevated",
            "composition": "layered composition, product hero, supporting props create context",
            "tags": ["vanity", "morning routine", "flat lay", "lifestyle", "authentic"],
        },
        {
            "name": "Water Splash Freshness",
            "scene": "bottle hovering above clear rippling water, frozen mid-air droplets and splashes",
            "lighting": "high-speed studio strobe backlight through water, crisp highlights",
            "style": "high-speed commercial product photography, Rankin style, dynamic action",
            "mood": "energetic, refreshing, pure, hydrating",
            "camera_angle": "eye-level with slight tilt for dynamism",
            "composition": "centered hero, symmetrical circular ripple pattern",
            "tags": ["splash", "water", "high speed", "fresh", "hydration"],
        },
    ]
    concepts = []
    for i, preset in enumerate(presets[:num_concepts]):
        c = dict(preset)
        c["color_palette"] = palette_presets[i % len(palette_presets)]
        concepts.append(c)
    return json.dumps({"concepts": concepts}, ensure_ascii=False)


def _score_response(concept_name: str) -> str:
    seed = sum(ord(ch) for ch in concept_name)
    rng = random.Random(seed)
    base_scores = {
        "Minimalist Studio":        (9.1, 8.8, 9.5, 9.2),
        "Golden Botanical Hour":    (8.4, 9.0, 8.2, 8.5),
        "Deep Blue Night Luxury":   (9.3, 7.9, 7.6, 8.1),
        "Morning Vanity Ritual":    (7.8, 9.2, 9.0, 8.3),
        "Water Splash Freshness":   (8.9, 8.1, 6.9, 8.7),
    }
    if concept_name in base_scores:
        c, r, re_, pq = base_scores[concept_name]
    else:
        c = rng.uniform(6.5, 9.0)
        r = rng.uniform(6.5, 9.0)
        re_ = rng.uniform(6.5, 9.0)
        pq = rng.uniform(6.5, 9.0)
    jitter = lambda v: max(1.0, min(10.0, v + rng.uniform(-0.3, 0.3)))
    c, r, re_, pq = jitter(c), jitter(r), jitter(re_), jitter(pq)
    avg = round((c + r + re_ + pq) / 4.0, 1)
    feedback = (
        "Strong concept with memorable visual hook. "
        "Strengths: great atmosphere and lighting plan. "
        "Add more concrete surface textures for best diffusion results."
    )
    return json.dumps(
        {
            "creativity": round(c, 1),
            "relevance": round(r, 1),
            "realizability": round(re_, 1),
            "prompt_quality": round(pq, 1),
            "avg": avg,
            "feedback": feedback,
        },
        ensure_ascii=False,
    )


def _refined_prompt_response(concept_name: str, lowered_prompt: str) -> str:
    """Mock refinement output for the refine_prompts node.

    Based on concept name we pick a plausible camera/lens/lighting string so
    downstream generation + evaluation see non-trivial prompts.
    """
    base_desc = "Premium vitamin C face serum, glass dropper bottle, rich golden-amber transparent liquid, matte black cap with brushed gold foil ring, thick clear soda-lime glass with visible rounded shoulder refractions, tiny silicone seal ring visible on collar"

    styles_by_keyword: list[tuple[str, str, str, list[str]]] = [
        ("minimalist", "white seamless studio backdrop, single pale travertine cylindrical pedestal",
         "soft diffused Broncolor Para 133 key light 45deg top-left, thin cool L1.15 6500K rim light behind, white foamcore fill front-right, CRI 98+ TLCI 98",
         ["white seamless", "travertine", "studio lighting", "broncolor", "magazine cover quality", "Peter Lippmann"]),
        ("golden", "rough natural travertine slab, soft out-of-focus eucalyptus leaf bokeh edges, thin window blind shadow bars across lower third",
         "golden hour 3200K warm backlight through foliage, dappled leaf shadows, large white foamcore fill from front, CRI 95+",
         ["golden hour", "natural light", "botanical", "lifestyle editorial", "travertine", "Stephane Coutelle"]),
        ("night", "glossy nero marquina marble base, deep indigo silk velvet drape background, tiny champagne bokeh specks, faint hard reflection under bottle",
         "dramatic single Aputure 600d Pro spot 70deg overhead with snoot, soft LED panel fill 20% from front, cool CTB gel, CRI 97",
         ["noir cinematic", "velvet drape", "marble reflection", "luxury night", "moody spotlight", "Quentin Jones"]),
        ("vanity", "warm solid white oak vanity countertop grain, linen tea towel, brass perfume atomizer, half full porcelain coffee mug, tiny eucalyptus sprig",
         "north-facing soft overcast daylight 5600K through large window, sheer white linen sheer diffusion, negative fill black card left, CRI 99",
         ["morning routine", "oak wood", "linen", "lifestyle vanity", "natural north light", "editorial tableau"]),
        ("water", "clear calm rippling aqua water surface directly below, frozen suspended crystal water droplets and splash crown mid air around bottle neck",
         "Profoto B10X large octabox 120cm front 30deg up, 2x strip lights sides for water speculars, 8000K cool white, CRI 96+ high speed sync",
         ["splash freeze", "water beauty", "high speed sync", "clean aqua", "cosmetic splash", "Nick Knight style"]),
    ]

    matched_style = None
    for kw, scene, lighting, tags in styles_by_keyword:
        if kw in concept_name.lower():
            matched_style = (scene, lighting, tags)
            break
    if matched_style is None:
        matched_style = styles_by_keyword[0][1:]  # default minimalist
    scene, lighting, style_tags = matched_style

    camera_composition = (
        "Hasselblad X1D II 50C, Hasselblad XCD 80mm f/1.9 lens, leaf shutter 1/125, ISO 100, "
        "Phase One IQ4 150MP, focus stacked 3 shots on bottle cap and shoulder, "
        "eye-level straight-on tight close-up, rule of thirds product left third, negative space right and top for headline and CTA copy, "
        "product 32% of frame height, shallow depth of field f/2.8, background bokeh falloff, "
        "colour science Hasselblad Natural Colour Solution, no clipping whites, full DR 14.5 stops, "
        "high end commercial post, glossy glass speculars correctly shaped, no warped refractions, "
        "liquid meniscus curved top visible inside neck, subtle condensation micro droplets optional, "
        "magazine retouch, no diffusion blur or banding, 16-bit tiff, print-ready"
    )

    positive = (
        f"Commercial editorial advertising photography of luxury skincare product. "
        f"Hero hero hero: {base_desc}. "
        f"Scene: {scene}. "
        f"Lighting: {lighting}. "
        f"Camera/shot: {camera_composition} "
        f"Aesthetic: ultra high end retouch, glossy glass reflections crisp, no banding, magazine quality, "
        f"commercial product photography, perfect colour harmony, sharp focus plane, correct perspective"
    )

    negative = (
        "deformed bottle, melted glass, broken dropper, extra tube, warped proportions, "
        "melted gold foil ring, 6 fingered hand, extra limbs, bad perspective glass refraction, "
        "wrong reflection pattern, text inside liquid, banding, posterization, aliased edges, "
        "watermarks, captions, typography overlay, jpeg artifacts, over sharpen halos, "
        "floating objects, detached cap, bad shadows, uncanny valley, digital painting, CG render, "
        "cartoon, anime, illustration, sketch, low resolution, blurry, duplicate bottles, multiple labels, "
        "ugly typography, logo distorted"
    )

    # Seed-based prompt strength note
    seed = sum(ord(ch) for ch in concept_name) % 5
    notes_opts = [
        "High-quality prompt with concrete camera + lens + modifier stack; should produce top 10% FLUX.1 schnell results on first try.",
        "Strong diffusion prompt with well-defined lighting and depth cues; minor tweaking of modifier angle may improve glass speculars.",
        "Great prompt — well-specified product materials, scene geometry, colour science, and focus targets all match luxury category.",
        "Excellent on-spec prompt. Recommend generating twice with different seeds, select the image with best dropper + collar rendering.",
        "Very robust commercial photographer prompt. Negative list covers all common FLUX product-artifact failure modes.",
    ]

    return json.dumps(
        {
            "positive_prompt": positive.strip(),
            "negative_prompt": negative.strip(),
            "style_boost_tags": style_tags,
            "estimated_prompt_strength_notes": notes_opts[seed],
        },
        ensure_ascii=False,
    )


class MockLLMBackend(LLMBackend):
    """Fast offline mock for pipeline logic testing — no downloads, no RAM usage."""

    def __init__(self, latency_seconds: float = 0.25):
        self.latency = latency_seconds
        self._calls = 0
        print("🧪 Using MockLLMBackend (fast offline testing mode — no real LLM)")

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
        **extra,
    ) -> str:
        self._calls += 1
        time.sleep(self.latency)
        lowered = prompt.lower()
        # 1) 4D concept scoring request
        if "rate this concept" in lowered or "criteria" in lowered and "realizability" in lowered:
            m = re.search(r'"name"\s*:\s*"([^"]+)"', prompt)
            name = m.group(1) if m else "Unknown"
            return _score_response(name)
        # 2) Prompt refinement request (FLUX.1 positive/negative prompt engineering)
        if "refine" in lowered or ("positive_prompt" in lowered and "negative_prompt" in lowered) or "prompt engineer" in lowered:
            m = re.search(r'"name"\s*:\s*"([^"]+)"', prompt)
            concept_name = (m.group(1) if m else "Luxury Product").strip()
            return _refined_prompt_response(concept_name, lowered)
        # 3) Default: creative concept generation
        return _concepts_response(5)
