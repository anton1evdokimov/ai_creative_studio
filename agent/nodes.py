import json
from typing import Any

from models.diffusion.config import load_pipeline_config, load_diffusion_config
from models.llm.factory import create_llm, unload_llm
from models.llm.parser import parse_concepts, parse_concept_score
from models.diffusion.generator import get_flux_generator, reset_flux_generator
from models.diffusion.factory import unload_image_backend
from models.vlm.analyzer import create_product_analyzer, reset_product_analyzer
from models.vlm.factory import unload_vlm
from models.ranking.clip import create_image_scorer, reset_image_scorer
from models.ranking.quality import analyze_quality
from agent.refine_prompts import create_prompt_refiner, reset_prompt_refiner
from schemas.creative import ScoredConcept

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = create_llm()
    return _llm


def _release_mlx():
    import gc

    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass


def _unload_vlm_weights():
    reset_product_analyzer()
    reset_image_scorer()
    unload_vlm()
    _release_mlx()
    print("   ♻️  Unloaded VLM")


def _unload_llm_weights():
    global _llm
    _llm = None
    reset_prompt_refiner()
    unload_llm()
    _release_mlx()
    print("   ♻️  Unloaded LLM")


def _unload_flux_weights():
    reset_flux_generator()
    unload_image_backend()
    _release_mlx()
    print("   ♻️  Unloaded FLUX")


def analyze_product(state):
    """Stage 1: use the shared VLM to deeply analyze the user's product photo."""
    print("🔎 [1/7] Analyzing product photo with VLM...")
    desc = state.get("product_description", "") or ""
    img = state.get("product_image", "") or ""

    analyzer = create_product_analyzer()
    analysis_obj = analyzer.analyze(
        image_path=img or "",   # empty string triggers safe text-only fallback
        user_description=desc,
        fallback_if_missing_image=True,
    )

    # legacy dict view
    analysis_dict: dict[str, Any] = {}
    try:
        analysis_dict = analysis_obj.model_dump(mode="json")
    except Exception:
        analysis_dict = {
            "category": analysis_obj.category,
            "product_type": analysis_obj.product_type,
            "product_name": analysis_obj.product_name,
            "materials": analysis_obj.materials,
            "colors": analysis_obj.colors,
            "color_palette_hex": analysis_obj.color_palette_hex,
            "shape": analysis_obj.shape,
            "size": analysis_obj.size,
            "luxury_level": analysis_obj.luxury_level,
            "brand_visual_cues": analysis_obj.brand_visual_cues,
            "key_visual_features": analysis_obj.key_visual_features,
            "packaging_type": analysis_obj.packaging_type,
            "extracted_text_ocr": analysis_obj.extracted_text_ocr,
            "visual_caption": analysis_obj.visual_caption,
            "suggested_target_audience": analysis_obj.suggested_target_audience,
        }
    # add convenience keys older code may expect
    analysis_dict.setdefault("category", analysis_obj.category or "consumer goods")
    analysis_dict.setdefault("product", analysis_obj.product_name or analysis_obj.product_type or desc or "product")
    analysis_dict.setdefault("style", f"{analysis_obj.luxury_level or 'mid-range'} commercial photography")
    analysis_dict.setdefault("target_audience", analysis_obj.suggested_target_audience or "")
    analysis_dict["product_image"] = img

    inferred = " ".join(
        part for part in (
            analysis_obj.product_name,
            analysis_obj.product_type,
            analysis_obj.visual_caption,
        ) if part
    ).strip()
    if not desc:
        desc = inferred or "retail product"
        print(f"   📝 No user description — using VLM read: {desc[:140]}{'…' if len(desc) > 140 else ''}")
    state["product_description"] = desc

    print(f"   🏷️  Category: {analysis_obj.category or '—'}  |  Type: {analysis_obj.product_type or '—'}")
    print(f"   💎 Luxury level: {analysis_obj.luxury_level}  |  Materials: {', '.join(analysis_obj.materials[:4]) or '—'}")
    if analysis_obj.colors:
        print(f"   🎨 Dominant colours: {', '.join(analysis_obj.colors[:4])}")
    if analysis_obj.visual_caption:
        cap = analysis_obj.visual_caption
        print(f"   📷 Caption: {cap[:160]}{'…' if len(cap) > 160 else ''}")

    state["product_analysis_obj"] = analysis_obj
    state["product_analysis"] = analysis_dict
    state["retry_count"] = int(state.get("retry_count", 0) or 0)
    _unload_vlm_weights()
    return state


def create_concepts(state):
    llm = _get_llm()
    pipe_cfg = load_pipeline_config()
    num_concepts = int(pipe_cfg["num_concepts"])
    print(f"💡 [2/7] Generating {num_concepts} diverse creative concepts...")

    try:
        product_info = json.dumps(state["product_analysis"], ensure_ascii=False)
    except Exception:
        product_info = json.dumps({"product": state.get("product_description", "")})
    product_desc = state.get("product_description", "") or ""
    lux = ""
    pa = state.get("product_analysis") or {}
    if isinstance(pa, dict):
        lux = str(pa.get("luxury_level") or pa.get("style") or "")

    prompt = f"""You are a senior creative director for luxury brands.

PRODUCT:
Description: {product_desc}
Luxury / vibe: {lux}
Analysis: {product_info}

TASK: Generate exactly {num_concepts} diverse advertising creative concepts.
Each concept must be visually distinct (different scene, mood, camera angle, lighting direction, background material).

RESPOND ONLY AS VALID JSON — no markdown, no extra words:
{{"concepts": [
  {{
    "name": "short memorable title (3-5 words, English)",
    "scene": "where product is placed, full environment description + background materials",
    "lighting": "specific lighting (e.g. soft golden hour backlight + bounce, studio 3-point octabox)",
    "style": "photography style references, mention a famous commercial photographer if relevant",
    "mood": "emotional tone (serene, energetic, aspirational, intimate, bold...)",
    "camera_angle": "e.g. eye-level close-up, 45-degree flat lay, low angle hero, overhead product tableau",
    "color_palette": ["#hex", "color name", ... 3-5 items],
    "composition": "e.g. rule of thirds, centered symmetry, negative space right for copy, product 30% frame",
    "tags": ["keyword1", "keyword2", "photography tag", ... 4-8 items]
  }}
]}}
"""

    response = llm.generate(prompt)
    print(f"   LLM response received ({len(response)} chars)")

    parsed = parse_concepts(response)
    concepts = parsed.concepts
    if not concepts:
        print("   ⚠️ No concepts parsed, using fallback concepts")
        from schemas.creative import CreativeConcept
        concepts = [
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
                scene="product on natural travertine slab, soft green eucalyptus leaves bokeh around edges, thin shadow bars from window blinds",
                lighting="golden hour warm sunlight 35deg backlight dappled leaf shadows, white foamcore fill",
                style="organic lifestyle editorial, natural light photography",
                mood="warm, fresh, natural, aspirational, calm",
                camera_angle="45-degree flat lay overhead, medium shot",
                color_palette=["#D4A574", "#8B9A6B", "#F7E7CE", "#E6B325"],
                composition="rule of thirds, product upper-left, negative space bottom-right for headline",
                tags=["lifestyle", "natural", "golden hour", "botanical", "outdoor", "editorial"],
            ),
        ]

    print(f"   ✅ Got {len(concepts)} concepts")
    for i, c in enumerate(concepts):
        print(f"      [{i+1}] {c.name}  —  {c.scene[:70]}")

    state["creative_concepts"] = concepts
    return state


def score_prompts(state):
    llm = _get_llm()
    pipe_cfg = load_pipeline_config()
    top_k = int(pipe_cfg["top_k_concepts"])
    concepts = state["creative_concepts"]
    print(f"⭐ [3/7] Scoring {len(concepts)} concepts → selecting top-{top_k}...")

    try:
        product_info = json.dumps(state["product_analysis"], ensure_ascii=False)
    except Exception:
        product_info = "{}"
    scored = []

    for idx, concept in enumerate(concepts):
        concept_json = concept.model_dump_json(ensure_ascii=False)
        prompt = f"""You are a prompt quality rater for FLUX.1 image diffusion.

PRODUCT CONTEXT: {product_info}

CONCEPT TO RATE: {concept_json}

Rate this concept on 4 criteria (1.0 worst — 10.0 best):
1. creativity: originality, would this stand out in a feed vs. competitor ads?
2. relevance: does the vibe / scene / styling match the product category and luxury level?
3. realizability: can a diffusion model actually draw this? no impossible physics, no water / fire interactions that always break.
4. prompt_quality: are the visual details concrete and translateable into a strong diffusion prompt?

RESPOND ONLY AS VALID JSON:
{{"creativity": N, "relevance": N, "realizability": N, "prompt_quality": N, "avg": N, "feedback": "very short 1-sentence comment"}}
"""

        response = llm.generate(prompt)
        score = parse_concept_score(response)
        scored.append(ScoredConcept(concept=concept, score=score))
        print(
            f"   [{idx+1}] {concept.name[:30]:<30} | "
            f"avg={score.avg:4.1f}  "
            f"(C={score.creativity:.0f} R={score.relevance:.0f} "
            f"Re={score.realizability:.0f} PQ={score.prompt_quality:.0f})"
            f"{'  ⭐' if score.avg >= 8.5 else ''}"
        )

    scored.sort(key=lambda s: s.score.avg, reverse=True)
    winners = scored[:top_k]
    print(f"\n   🏆 Selected top-{top_k} for diffusion:")
    for rank, s in enumerate(winners, 1):
        print(f"      #{rank}  {s.concept.name}  avg={s.score.avg:.1f}  — {s.score.feedback}")

    state["scored_concepts"] = scored
    state["ranked_concepts"] = [s.concept for s in winners]
    return state


def refine_prompts(state):
    """NEW STAGE 4: LLM prompt engineer produces professional FLUX.1 positive/negative prompts."""
    pipe_cfg = load_pipeline_config()
    top_k = int(pipe_cfg["top_k_concepts"])
    winners = state.get("ranked_concepts") or state.get("creative_concepts", [])
    print(f"✍️  [4/7] Refining top-{len(winners)} concepts → FLUX.1 prompts...")

    refiner = create_prompt_refiner()
    analysis_obj = state.get("product_analysis_obj") or None
    refined = []
    for idx, concept in enumerate(winners):
        r = refiner.refine(concept, product_analysis=analysis_obj)
        refined.append(r)
        print(f"   [{idx+1}] {r.concept_name}")
        print(f"       +ve ({len(r.positive_prompt)} chars): {r.positive_prompt[:110]}…")
        if r.style_boost_tags:
            print(f"       style boost: {', '.join(r.style_boost_tags[:4])}")

    state["refined_prompts"] = refined
    _unload_llm_weights()
    return state


def generate_images(state):
    print("🎨 [5/7] Generating images...")
    generator = get_flux_generator()
    concepts = state.get("ranked_concepts") or state.get("creative_concepts", [])
    refined = state.get("refined_prompts") or None
    pa = state.get("product_analysis") or {}
    product_desc = state.get("product_description", "") or ""
    if not product_desc and isinstance(pa, dict):
        product_desc = str(pa.get("visual_caption") or pa.get("product") or "")

    results = generator.generate_concepts(
        concepts,
        product_description=product_desc,
        refined_prompts=refined,
    )

    state["generated_images"] = [item.image_path for item in results]
    state["generation_results"] = results
    _unload_flux_weights()
    return state


def evaluate_images(state):
    """Stage 6: VLM-as-judge scores each image on 6 dimensions + cheap heuristics."""
    print("📊 [6/7] Evaluating generated images (VLM + quality heuristics)...")
    diff_cfg = load_diffusion_config()
    target_w = int(diff_cfg.get("width", 832))
    target_h = int(diff_cfg.get("height", 832))

    pa = state.get("product_analysis") or {}
    try:
        context_json = json.dumps(pa, ensure_ascii=False)[:600]
    except Exception:
        context_json = "luxury retail product advertisement"

    scorer = create_image_scorer()

    # Build prompt-per-image from generation_results (prefer refined prompts)
    generation_results = state.get("generation_results") or []
    generated_images = state.get("generated_images") or []

    def _prompt_for(i: int, path: str) -> str:
        if i < len(generation_results):
            gr = generation_results[i]
            if getattr(gr, "refined_prompt", None) is not None:
                return gr.refined_prompt.positive_prompt
            return getattr(gr, "prompt", "") or ""
        return ""

    typed_results = []
    legacy_dicts = []
    best_path: str | None = None
    best_score: float = -1.0

    for idx, path in enumerate(generated_images):
        prompt = _prompt_for(idx, path)
        print(f"   [{idx+1}/{len(generated_images)}] scoring {path.split('/')[-1]}")
        # 6D VLM scoring (prompt alignment, aesthetic, product accuracy, realism, brand_fit, overall)
        typed = scorer.score_image_result(
            image_path=path,
            prompt_text=prompt,
            context_text=context_json,
        )
        # Cheap deterministic checks (opens? resolution? blur? compression?)
        q = analyze_quality(path, expected_width=target_w, expected_height=target_h)
        if typed.vlm_scores:
            final = round(0.75 * typed.vlm_scores.overall + 0.25 * q["quality_factor"], 3)
        else:
            final = q["quality_factor"]
        typed.score = final  # override with blended final
        typed_results.append(typed)

        legacy_dicts.append({
            "image": path,
            "score": final,
            "clip": typed.clip_score or (typed.vlm_scores.prompt_alignment if typed.vlm_scores else None),
            "quality": typed.quality_score or q["quality_factor"],
            "blur": q.get("blur_score"),
            "meets_resolution": q.get("meets_resolution"),
            "size_bytes": q.get("size_bytes"),
            "vlm_overall": typed.vlm_scores.overall if typed.vlm_scores else None,
            "feedback": typed.vlm_scores.feedback if typed.vlm_scores else "",
            "quality_notes": q.get("notes", []),
        })
        if final > best_score:
            best_score = final
            best_path = path

        # compact one-line summary
        sc = typed.vlm_scores
        if sc is not None:
            print(
                f"       VLM: prompt={sc.prompt_alignment:.2f}  "
                f"aesth={sc.aesthetic_quality:.2f}  "
                f"product={sc.product_accuracy:.2f}  "
                f"realism={sc.realism:.2f}  →  OVERALL={final:.3f}"
            )
            if sc.feedback:
                print(f"       💬 {sc.feedback[:140]}")
        else:
            print(f"       heuristics only → final={final:.3f}")

    state["evaluation_typed"] = typed_results
    state["evaluation_results"] = legacy_dicts
    if best_path:
        state["best_image"] = best_path
        print(f"\n   🏆 Best image: {best_path.split('/')[-1]}   (final={best_score:.3f})")
    _unload_vlm_weights()
    return state


def improve_prompt(state):
    print("🔄 Improving prompts before retry...")
    state["retry_count"] = int(state.get("retry_count", 0) or 0) + 1
    # also clear refined_prompts so refine_prompts node re-runs
    if "refined_prompts" in state:
        del state["refined_prompts"]
    enhanced = []
    for concept in state.get("ranked_concepts") or state.get("creative_concepts", []):
        updates = {
            "style": (concept.style + ", photorealistic, ultra detailed, 8k, sharp focus, accurate glass refraction").strip(","),
            "lighting": (concept.lighting + ", high CRI TLCI 98+, soft shadows, cinematic, correct product reflections").strip(","),
            "tags": list(dict.fromkeys((concept.tags or []) + ["photorealistic", "high detail", "professional", "print ready"])),
        }
        enhanced.append(concept.model_copy(update=updates))
    state["ranked_concepts"] = enhanced
    state["creative_concepts"] = enhanced
    return state


def quality_router(state):
    pipe_cfg = load_pipeline_config()
    threshold = float(pipe_cfg["quality_threshold"])
    max_retries = int(pipe_cfg["max_retries"])
    retry_count = int(state.get("retry_count", 0) or 0)

    results = state.get("evaluation_results") or []
    if not results:
        return "end"

    best_score = max(float(item.get("score", 0.0) or 0.0) for item in results)

    if threshold <= 0.0:
        return "end"
    if best_score >= threshold:
        print(f"   ✅ Best score {best_score:.3f} >= threshold {threshold:.3f} → ending")
        return "end"
    if retry_count >= max_retries:
        print(f"   ⏹️  Best score {best_score:.3f} < {threshold:.3f} but out of retries ({retry_count}/{max_retries}) → ending")
        return "end"
    print(f"   🔁 Best score {best_score:.3f} < {threshold:.3f} → re-running with improved prompts (attempt {retry_count+1}/{max_retries})")
    return "improve"


def generate_video(state):
    print("🎬 Generating video placeholders...")
    state["generated_videos"] = [f"{i}_video.mp4" for i in state.get("generated_images", [])]
    return state
