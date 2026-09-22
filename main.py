import sys
import time
import traceback
from pathlib import Path

# ------------------------------------------------------------------
# Apple Silicon MLX requires Python 3.10+.
# Stop early with a clear message if someone runs `python3` (3.9).
# ------------------------------------------------------------------
_MIN_PY = (3, 10)
if sys.version_info < _MIN_PY:
    print(
        "❌ Python",
        ".".join(map(str, sys.version_info[:3])),
        "is too old. MLX requires Python >= 3.10.",
    )
    print("   Run with:  python3.11 main.py")
    print("   (or create a venv with `python3.11 -m venv .venv && source .venv/bin/activate`)")
    sys.exit(2)

from agent.graph import build_graph


BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║   🎨 AI Creative Studio  •  LangGraph + MLX + FLUX.1        ║
║   + VLM product analyzer + 6D image scorer + prompt refine  ║
║   Optimized for Apple Silicon M1 8GB RAM                    ║
╚══════════════════════════════════════════════════════════════╝
"""


def _fmt_list(lst: list[str], n: int = 6) -> str:
    if not lst:
        return "—"
    head = lst[:n]
    tail = "…+{}".format(len(lst) - n) if len(lst) > n else ""
    return ", ".join(head) + tail


def main() -> int:
    print(BANNER)

    t0 = time.time()

    input_data = {
        "product_image": "data/input/serum.jpeg",
        "retry_count": 0,
    }

    try:
        print("🛠️  Building LangGraph pipeline (7 stages)...")
        graph = build_graph()
        print("✅ Pipeline compiled successfully")
        print()

        result = graph.invoke(input_data)

        dt = time.time() - t0
        print()
        print("=" * 70)
        print(f"🎉 Pipeline finished in {dt:,.1f}s  ·  stages: analyze→concepts→scoring→refine→generate→evaluate")
        print("=" * 70)

        # ------------------------------------------------------------------
        # STAGE 1 SUMMARY: VLM product analysis
        # ------------------------------------------------------------------
        pa_obj = result.get("product_analysis_obj")
        pa = result.get("product_analysis") or {}
        if pa_obj is not None or pa:
            print("\n🏷️  [Stage 1] VLM Product Analysis")
            if pa_obj is not None:
                print(f"   Category / type : {pa_obj.category or '—'} / {pa_obj.product_type or '—'}")
                print(f"   Product name    : {pa_obj.product_name or '—'}")
                print(f"   Luxury level    : {pa_obj.luxury_level or '—'}")
                print(f"   Shape / size    : {pa_obj.shape or '—'} · {pa_obj.size or '—'}")
                print(f"   Materials       : {_fmt_list(pa_obj.materials)}")
                print(f"   Colours / HEX   : {_fmt_list(pa_obj.colors)}  {_fmt_list(pa_obj.color_palette_hex)}")
                print(f"   Key visual feats: {_fmt_list(pa_obj.key_visual_features)}")
                print(f"   Audience        : {pa_obj.suggested_target_audience[:120] if pa_obj.suggested_target_audience else '—'}")
                if pa_obj.visual_caption:
                    print(f"   📷 Caption      : {pa_obj.visual_caption[:220]}")
            else:
                print(f"   Dict summary    : {pa}")

        # ------------------------------------------------------------------
        # STAGES 2-3 SUMMARY: concepts + 4D scoring
        # ------------------------------------------------------------------
        scored = result.get("scored_concepts", [])
        if scored:
            print("\n⭐ [Stages 2-3] Creative concepts — 4D ranked (top first)")
            header = f"  {'#':>2}  {'avg':>4}  C  R  Re PQ  concept"
            print(header)
            print("  " + "-" * (len(header) - 2))
            for rank, s in enumerate(scored, 1):
                mark = "🏆" if rank <= len(result.get("ranked_concepts") or []) else "  "
                print(
                    f"  {rank:>2} {mark} {s.score.avg:4.1f}  "
                    f"{s.score.creativity:2.0f} {s.score.relevance:2.0f} "
                    f"{s.score.realizability:2.0f} {s.score.prompt_quality:2.0f}   "
                    f"{s.concept.name}"
                )
                if s.score.feedback:
                    print(f"            💬 {s.score.feedback[:140]}")

        # ------------------------------------------------------------------
        # STAGE 4: Refined prompts
        # ------------------------------------------------------------------
        refined = result.get("refined_prompts") or []
        if refined:
            print("\n✍️  [Stage 4] Refined FLUX.1 prompts (positive + negative)")
            for i, r in enumerate(refined, 1):
                print(f"  [{i}] {r.concept_name}")
                print(f"      +ve ({len(r.positive_prompt)} chars): {r.positive_prompt[:160]}…")
                if r.negative_prompt:
                    print(f"      -ve ({len(r.negative_prompt)} chars): {r.negative_prompt[:120]}…")
                if r.style_boost_tags:
                    print(f"      style boost: {', '.join(r.style_boost_tags[:6])}")
                if r.estimated_prompt_strength_notes:
                    print(f"      strength   : {r.estimated_prompt_strength_notes[:140]}")

        # ------------------------------------------------------------------
        # STAGE 5: Generated images
        # ------------------------------------------------------------------
        images = result.get("generated_images") or []
        if images:
            print("\n🎨 [Stage 5] Generated images")
            for p in images:
                path = Path(p)
                try:
                    size_kb = path.stat().st_size / 1024 if path.exists() else 0
                except OSError:
                    size_kb = 0
                print(f"  • {p}  ({size_kb:,.0f} KB)")
        else:
            print("\n⚠️  No images were generated.")

        # ------------------------------------------------------------------
        # STAGE 6: Evaluation (VLM 6D + heuristics)
        # ------------------------------------------------------------------
        evals_typed = result.get("evaluation_typed") or []
        evals = result.get("evaluation_results") or []
        if evals_typed or evals:
            print("\n📊 [Stage 6] Image Evaluation (VLM-as-judge + quality heuristics)")
            if evals_typed:
                hdr = f"  {'#':>2}  final  prompt aesth prodc realis brand     file"
                print(hdr)
                print("  " + "-" * (len(hdr) - 2))
                for idx, e in enumerate(evals_typed, 1):
                    s = e.vlm_scores
                    name = Path(e.image_path).name
                    mark = "🏆" if (e.image_path == result.get("best_image")) else "  "
                    if s is not None:
                        print(
                            f"  {idx:>2} {mark} {e.score:5.3f}  "
                            f"{s.prompt_alignment:4.2f} {s.aesthetic_quality:4.2f} "
                            f"{s.product_accuracy:4.2f} {s.realism:4.2f} {s.brand_fit:4.2f}   {name}"
                        )
                        if s.feedback:
                            print(f"           💬 {s.feedback[:160]}")
                    else:
                        print(f"  {idx:>2} {mark} {e.score:5.3f}  — heuristics only — {name}")
            elif evals:
                for e in evals:
                    print(f"  • {e.get('image')}  score={e.get('score')}")

        best = result.get("best_image")
        if best:
            print(f"\n🏆 BEST IMAGE: {best}")

        retries = int(result.get("retry_count") or 0)
        if retries:
            print(f"   Retries used: {retries}")

        return 0

    except KeyboardInterrupt:
        print("\n⏹️  Cancelled by user.")
        return 130
    except Exception as exc:
        dt = time.time() - t0
        print(f"\n💥 ERROR after {dt:,.1f}s: {type(exc).__name__}: {exc}")
        print("--- traceback ---")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

