from langgraph.graph import StateGraph, END

from agent.state import PipelineState

from agent.nodes import (
    analyze_product,
    create_concepts,
    score_prompts,
    refine_prompts,
    generate_images,
    generate_video,
    evaluate_images,
    improve_prompt,
    quality_router,
    human_gate,
    human_router,
)

_saver_keep = None


def _redis_saver(url: str):
    from langgraph.checkpoint.redis import RedisSaver

    saver = RedisSaver(redis_url=url)
    if hasattr(saver, "setup"):
        saver.setup()
    return saver


def _sqlite_saver(db_path: str):
    from pathlib import Path
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)


def _checkpointer(db_path: str | None = None):
    """REDIS_URL → Redis; else sqlite file; else memory."""
    global _saver_keep
    import os

    url = (os.environ.get("REDIS_URL") or os.environ.get("AICS_REDIS_URL") or "").strip()
    if url:
        try:
            _saver_keep = _redis_saver(url)
            print(f"   LangGraph checkpoint: redis ({url})")
            return _saver_keep
        except Exception as extra:
            print(f"⚠️  Redis checkpointer failed ({type(extra).__name__}: {extra})")
    if db_path:
        try:
            _saver_keep = _sqlite_saver(db_path)
            print(f"   LangGraph checkpoint: sqlite ({db_path})")
            return _saver_keep
        except Exception as extra:
            print(f"⚠️  sqlite checkpointer unavailable ({type(extra).__name__}: {extra})")
    try:
        from langgraph.checkpoint.memory import MemorySaver

        _saver_keep = MemorySaver()
        print("   LangGraph checkpoint: memory")
        return _saver_keep
    except Exception:
        return None


def build_graph(checkpointer=None, db_path: str | None = None):
    workflow = StateGraph(PipelineState)

    workflow.add_node("entry", lambda state: state)
    workflow.add_node("analysis", analyze_product)
    workflow.add_node("planning", create_concepts)
    workflow.add_node("scoring", score_prompts)
    workflow.add_node("refine_prompts", refine_prompts)
    workflow.add_node("generation", generate_images)
    workflow.add_node("evaluation", evaluate_images)
    workflow.add_node("human_gate", human_gate)
    workflow.add_node("video_generation", generate_video)
    workflow.add_node("improve_prompt", improve_prompt)

    workflow.set_entry_point("entry")
    workflow.add_conditional_edges(
        "entry",
        lambda s: "generation" if s.get("want_direct") else "analysis",
        {"generation": "generation", "analysis": "analysis"},
    )
    workflow.add_edge("analysis", "planning")
    workflow.add_edge("planning", "scoring")
    workflow.add_edge("scoring", "refine_prompts")
    workflow.add_edge("refine_prompts", "generation")
    workflow.add_edge("generation", "evaluation")
    workflow.add_edge("evaluation", "human_gate")
    workflow.add_conditional_edges(
        "human_gate",
        human_router,
        {
            "continue": "video_generation",
            "retry": "improve_prompt",
            "direct_retry": "generation",
            "end": END,
        },
    )
    workflow.add_conditional_edges(
        "video_generation",
        quality_router,
        {"end": END, "improve": "improve_prompt"},
    )
    workflow.add_edge("improve_prompt", "refine_prompts")

    saver = checkpointer if checkpointer is not None else _checkpointer(db_path)
    if saver is not None:
        return workflow.compile(checkpointer=saver)
    return workflow.compile()
