from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

def generate_text(prompt: str) -> str:
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt],
    )
    # Extract the text safely
    text = ""
    for part in response.candidates[0].content.parts:
        if part.text:
            text += part.text
    return text.strip()

