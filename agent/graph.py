from langgraph.graph import StateGraph, END

from agent.state import PipelineState

from agent.nodes import (
    analyze_product,
    create_concepts,
    generate_images,
    generate_video,
    evaluate_images,
    improve_prompt,
    quality_router,
)


def build_graph():

    workflow = StateGraph(PipelineState)

    workflow.add_node(
        "analysis",
        analyze_product
    )

    workflow.add_node(
        "planning",
        create_concepts
    )

    workflow.add_node(
        "generation",
        generate_images
    )
    
    workflow.add_node(
    "video_generation",
    generate_video
    )   

    workflow.add_node(
        "evaluation",
        evaluate_images
    )
    
    workflow.add_node(
        "improve_prompt",
        improve_prompt
    )

    workflow.set_entry_point(
        "analysis"
    )

    workflow.add_edge(
        "analysis",
        "planning"
    )

    workflow.add_edge(
        "planning",
        "generation"
    )

    workflow.add_edge(
        "generation",
        "evaluation"
    )

    workflow.add_conditional_edges(
    "evaluation",
    quality_router,
    {
        "end": END,
        "improve": "improve_prompt"
    }
    )
    
    workflow.add_edge(
    "improve_prompt",
    "generation"
    )  

    return workflow.compile()
