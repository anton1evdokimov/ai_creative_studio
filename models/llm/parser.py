import json

from schemas.creative import CreativeConcepts


def parse_concepts(response: str):

    data = json.loads(response)

    return CreativeConcepts(**data)