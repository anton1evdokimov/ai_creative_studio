# LangGraph nodes: analysis, planning, generation, evaluation
from models.llm.factory import create_llm
from models.llm.parser import parse_concepts

llm = create_llm()
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

def improve_prompt(state):

    print("🔄 Improving prompt")


    state["creative_concepts"] = [
        concept.copy(update={
            "style": concept.style + ", more realistic"
        })
        for concept in state["creative_concepts"]
    ]


    return state

def quality_router(state):

    results = state["evaluation_results"]

    best_score = max(
        item["score"]
        for item in results
    )

    if best_score >= 0.8:
        return "end"

    return "improve"

def generate_video(state):

    print("🎬 Generate video")

    images = state["generated_images"]

    videos = []

    for image in images:
        video_path = f"{image}_video.mp4"

        # пока заглушка
        videos.append(video_path)


    state["generated_videos"] = videos

    return state