from fastapi import APIRouter
from app.services.agent.gemini_models import generate_text
from app.services.agent.image_generator import generate_image
from fastapi.responses import StreamingResponse
from io import BytesIO

router = APIRouter()

@router.get("/")
def read_root():
    return {"message": "ok"}

@router.get("/joke")
def tell_joke():
    joke_prompt = "tell me a joke about my colleague 'BFCAI'"
    joke = generate_text(joke_prompt)
    return {"joke": joke}

@router.get("/describe/{word}")
def describe_word(word: str):
    prompt = f"Describe the meaning of the English word '{word}' in one clear sentence."
    description = generate_text(prompt)
    return {"word": word, "description": description}

@router.get("/type/{word}")
def get_word_type(word: str):
    prompt = f"Tell me the part of speech of the word '{word}' (e.g., noun, verb, adjective)."
    word_type = generate_text(prompt)
    return {"word": word, "type": word_type}

@router.get("/example/{word}")
def get_word_example(word: str):
    prompt = f"Write a simple example sentence using the word '{word}'."
    example = generate_text(prompt)
    return {"word": word, "example": example}

@router.get("/image/{word}")
def get_word_image(word: str):
    prompt = f"Create a realistic image representing the word '{word}'."
    image = generate_image(prompt)
    if image is None:
        return {"error": "Could not generate image"}

    img_bytes = BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    return StreamingResponse(img_bytes, media_type="image/png")
