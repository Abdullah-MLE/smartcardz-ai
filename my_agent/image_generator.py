from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import uuid
import os

client = genai.Client()

def generate_image(prompt: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[prompt],
    )

    os.makedirs("generated_images", exist_ok=True)
    file_name = f"generated_images/{uuid.uuid4().hex}.png"

    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            image = Image.open(BytesIO(part.inline_data.data))
            image.save(file_name)
            return file_name

    return "No image generated"
