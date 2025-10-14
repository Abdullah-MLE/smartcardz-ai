# import os
# from dotenv import load_dotenv
# import google.generativeai as genai

# # Load environment variables
# load_dotenv()

# # Configure the generative AI client
# if "GOOGLE_API_KEY" in os.environ:
#     genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
# else:
#     raise ValueError("GOOGLE_API_KEY not found in environment variables.")

# def generate_text(prompt: str):
#     model = genai.GenerativeModel('gemini-2.5-flash')
#     response = model.generate_content(prompt)
#     return response.text






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

