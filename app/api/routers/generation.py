from fastapi import APIRouter, Request
from app.services.agent.gemini_models import generate_text
from app.services.agent.image_generator import generate_image, save_image
import base64
from io import BytesIO

router = APIRouter()

@router.get("/generate/{word}")
def generate_all(word: str, request: Request):  # <-- أضف request هنا
    description_prompt = f"Describe the meaning of the English word '{word}' in one clear sentence."
    type_prompt = f"Tell me the part of speech of the word '{word}' (e.g., noun, verb, adjective)."
    example_prompt = f"Write a simple example sentence using the word '{word}'."
    image_prompt = f"Create a realistic image representing the word '{word}'."

    description = generate_text(description_prompt)
    word_type = generate_text(type_prompt)
    example = generate_text(example_prompt)
    image = generate_image(image_prompt)

    image_base64 = None
    if image:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "word": word,
        "definition": description,
        "type": word_type,
        "example": example,
        "image_base64": image_base64,
    }
