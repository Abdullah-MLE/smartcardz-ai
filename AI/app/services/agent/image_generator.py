from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import uuid
import os

client = genai.Client()

def generate_image(prompt: str):
    # response = client.models.generate_content(
    #     model="gemini-2.5-flash-image",
    #     contents=[prompt],
    # )

    # for part in response.candidates[0].content.parts:
    #     if part.inline_data is not None:
    #         image = Image.open(BytesIO(part.inline_data.data))
    #         return image
    return None

def save_image(image: Image, folder: str) -> str:
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join(folder, filename)
    image.save(filepath)
    return filepath