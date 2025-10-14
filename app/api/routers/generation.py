from fastapi import APIRouter
from app.services.agent.gemini_models import generate_text
from app.services.agent.image_generator import generate_image, save_image

router = APIRouter()

@router.get("/generate/{word}")
def generate_all(word: str):
    description_prompt = f"Describe the meaning of the English word '{word}' in one clear sentence."
    type_prompt = f"Tell me the part of speech of the word '{word}' (e.g., noun, verb, adjective)."
    example_prompt = f"Write a simple example sentence using the word '{word}'."
    image_prompt = f"Create a realistic image representing the word '{word}'."

    description = generate_text(description_prompt)
    word_type = generate_text(type_prompt)
    example = generate_text(example_prompt)
    image = generate_image(image_prompt)

    image_path = None
    if image:
        image_path = save_image(image, "app/generated_images")

    return {
        "word": word,
        "description": description,
        "type": word_type,
        "example": example,
        "image_path": image_path,
    }