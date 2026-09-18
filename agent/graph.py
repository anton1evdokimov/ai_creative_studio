from langgraph.graph import END, StateGraph

from agent.nodes import (
    analyze_product,
    create_concepts,
    evaluate_images,
    generate_images,
)
from agent.state import CreativeState


def build_graph():

    workflow = StateGraph(
        CreativeState
    )

    workflow.add_node(
        "analyze",
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
        "evaluation",
        evaluate_images
    )

    workflow.set_entry_point(
        "analyze"
    )

    workflow.add_edge(
        "analyze",
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

    workflow.add_edge(
        "evaluation",
        END
    )

    return workflow.compile()
