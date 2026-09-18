# LangGraph nodes: analysis, planning, generation, evaluation
from models.llm import llm
from models.llm.parser import parse_concepts


# Анализ товара
def analyze_product(state):

    print("🔎 Analyzing product...")

    analysis = {
        "category": "premium cosmetics",
        "product": "face serum",
        "style": "luxury"
    }

    state["product_analysis"] = analysis

    return state

# Creative planner

def create_concepts(state):

    print("💡 Creating concepts...")


    prompt = f"""
        You are a creative director for a premium cosmetics brand.

        Product information:

        {state["product_analysis"]}

        Generate 3 advertising concepts.

        Return ONLY valid JSON:

        {{
        "concepts": [
        {{
        "name": "",
        "scene": "",
        "lighting": "",
        "style": ""
        }}
        ]
        }}
        """


    response = llm.generate(prompt)
    
    print("RAW LLM RESPONSE:")
    print(response)

    concepts = parse_concepts(response)


    state["creative_concepts"] = concepts.concepts


    return state

# Generation
def generate_images(state):

    print("🎨 Generating images...")


    images = [
        "generated/spa.png",
        "generated/nature.png"
    ]


    state["generated_images"] = images


    return state

# Evaluation
def evaluate_images(state):

    print("📊 Evaluating images...")


    results = [
        {
            "image": "generated/spa.png",
            "score": 0.87
        },
        {
            "image": "generated/nature.png",
            "score": 0.74
        }
    ]


    state["evaluation_results"] = results


    best = max(
        results,
        key=lambda x: x["score"]
    )


    state["best_image"] = best["image"]


    return state