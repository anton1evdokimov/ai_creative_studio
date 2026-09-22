from langgraph.graph import StateGraph, END

from agent.state import PipelineState

from agent.nodes import (
    analyze_product,
    create_concepts,
    score_prompts,
    refine_prompts,          # NEW: LLM prompt engineering for FLUX.1
    generate_images,
    generate_video,
    evaluate_images,
    improve_prompt,
    quality_router,
)


def build_graph():
    workflow = StateGraph(PipelineState)

    workflow.add_node("analysis", analyze_product)
    workflow.add_node("planning", create_concepts)
    workflow.add_node("scoring", score_prompts)           # Stage 3: ОЦЕНКА ПРОМПТОВ
    workflow.add_node("refine_prompts", refine_prompts)   # Stage 4: NEW — professional FLUX prompts
    workflow.add_node("generation", generate_images)
    workflow.add_node("video_generation", generate_video)
    workflow.add_node("evaluation", evaluate_images)
    workflow.add_node("improve_prompt", improve_prompt)

    workflow.set_entry_point("analysis")

    workflow.add_edge("analysis", "planning")
    workflow.add_edge("planning", "scoring")
    workflow.add_edge("scoring", "refine_prompts")        # ⭐ scoring → refine → generation
    workflow.add_edge("refine_prompts", "generation")
    workflow.add_edge("generation", "evaluation")

    workflow.add_conditional_edges(
        "evaluation",
        quality_router,
        {"end": END, "improve": "improve_prompt"},
    )
    # When we retry via improve_prompt we clear refined_prompts; so go BACK through refine_prompts
    # (rather than directly to generation). This way the LLM re-optimises the stronger concept.
    workflow.add_edge("improve_prompt", "refine_prompts")

    return workflow.compile()

