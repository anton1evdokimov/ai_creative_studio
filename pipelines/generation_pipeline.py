from models.diffusion.generator import FluxGenerator


def generate_images(concepts, product_description: str):
    generator = FluxGenerator()
    return generator.generate_concepts(concepts, product_description)
