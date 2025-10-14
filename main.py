from fastapi import FastAPI
from my_agent.gemini_models import generate_text
from my_agent.image_generator import generate_image

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "ok"}


@app.get("/joke")
def tell_joke():
    joke_prompt = "tell me a joke about my colleague 'BFCAI'"
    joke = generate_text(joke_prompt)
    return {"joke": joke}


@app.get("/describe/{word}")
def describe_word(word: str):
    prompt = f"Describe the meaning of the English word '{word}' in one clear sentence."
    description = generate_text(prompt)
    return {"word": word, "description": description}


@app.get("/type/{word}")
def get_word_type(word: str):
    prompt = f"Tell me the part of speech of the word '{word}' (e.g., noun, verb, adjective)."
    word_type = generate_text(prompt)
    return {"word": word, "type": word_type}


@app.get("/example/{word}")
def get_word_example(word: str):
    prompt = f"Write a simple example sentence using the word '{word}'."
    example = generate_text(prompt)
    return {"word": word, "example": example}


@app.get("/image/{word}")
def get_word_image(word: str):
    prompt = f"Create a realistic image representing the word '{word}'."
    image_path = generate_image(prompt)
    return {"word": word, "image_path": image_path}
