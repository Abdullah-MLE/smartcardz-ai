from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
import asyncio
from google import genai
from libs.GeminiWrapper.GeminiWrapper import GeminiWrapper
from libs.GeminiWrapper.models import InputParams, TextParams, ImageParams
from libs.FirebaseCRUD.FirebaseCRUD import FirebaseCRUD
from models import WordAgentResponse, WordAgentInput
from prompts import get_word_agent_system_prompt, get_word_agent_user_prompt
import io
from PIL import Image
import logging

load_dotenv()

app = FastAPI()

import os

def init_gemini_client():
    """Initialize and return a Gemini client (supports Vertex AI)."""
    project_id = os.environ.get("GCP_PROJECT_ID")
    location = os.environ.get("GCP_LOCATION", "us-central1")
    
    if project_id:
        # Using Vertex AI to consume GCP Free Credits
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location
        )
    else:
        # Fallback to AI Studio if no Project ID is provided
        client = genai.Client()
        
    return client

# Global Gemini Wrapper instance
gemini_client = init_gemini_client()
gemini_wrapper = GeminiWrapper(gemini_client)

# Initialize FirebaseCRUD
firebase_crud = FirebaseCRUD()

# -----------------------------------------------------------------------------
# Image Generation Helper
# -----------------------------------------------------------------------------
def _resize_image(image_bytes: bytes, max_dimension: Optional[int] = 400) -> bytes:
    """
    Resizes image bytes if dimensions exceed max_dimension, maintaining aspect ratio.
    Does NOT upscale.
    """
    if not max_dimension:
        return image_bytes

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
            if width <= max_dimension and height <= max_dimension:
                return image_bytes
                
            scale_factor = min(max_dimension / width, max_dimension / height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            fmt = img.format if img.format else 'JPEG'
            resized_img.save(output, format=fmt)
            return output.getvalue()
            
    except Exception as e:
        logging.getLogger("GeminiWrapper").warning(f"Image resizing failed: {e}")
        return image_bytes

def generate_post_image(image_prompt: str, gemini_wrapper: GeminiWrapper, userid: str = "None", card_id: str = "None") -> tuple[Optional[str], Optional[str]]:
    """Generates an image from a prompt, resizes it, and uploads to Firebase."""
    if not image_prompt:
        return None, None

    input_params = InputParams(
        prompt=image_prompt,
        model="gemini-2.5-flash-image"
    )
    image_params = ImageParams(output_image_aspect_ratio="1:1") # Defaulting to 1:1 as per common square posts
    
    result = gemini_wrapper.generate_image(input_params=input_params, image_params=image_params)
    
    if not result["success"]:
        print(f"\n--- IMAGE GENERATION FAILED ---")
        print(f"Error: {result.get('error')}")
        print(f"Raw Logs: {result.get('error_log')}")
        print(f"-------------------------------\n")
        return None, None
    
    image_bytes = result["content"]
    
    # Resize the image
    image_bytes = _resize_image(image_bytes, max_dimension=400)
    
    # Upload to Firebase
    if userid != "None" and card_id != "None":
        folder_path = f"post_images/{userid}/{card_id}"
        custom_fn = "ai.jpeg"
    else:
        folder_path = "post_images"
        custom_fn = None
        
    public_url, file_path = firebase_crud.upload_image(
        image_bytes, 
        folder=folder_path, 
        custom_filename=custom_fn
    )
    return public_url, file_path

# -----------------------------------------------------------------------------
# Main Logic
# -----------------------------------------------------------------------------
def process_word_sync(
    input_data: WordAgentInput,
    gemini_wrapper: GeminiWrapper
):
    # 1. Generate Text Content
    system_prompt = get_word_agent_system_prompt()
    user_prompt = get_word_agent_user_prompt(input_data)
    
    input_params = InputParams(
        prompt=user_prompt,
        system_instruction=system_prompt,
    )
    text_params = TextParams(
        response_schema=WordAgentResponse,
        response_mime_type='application/json',
    )
    
    result = gemini_wrapper.generate_text(input_params=input_params, text_params=text_params)
    
    if not result["success"]:
        return {"error": result.get("error")}

    word_data: WordAgentResponse = result["content"]

    # Convert Pydantic model to dict to allow modification
    response_dict = word_data.model_dump()

    # 2. Generate Image if valid and prompt exists
    image_url = None
    image_path = None
    if word_data.is_valid and word_data.image_prompt:
        image_url, image_path = generate_post_image(
            word_data.image_prompt, 
            gemini_wrapper, 
            userid=input_data.userid, 
            card_id=input_data.card_id
        )
        
    # 3. Store in Firebase Firestore
    db_data = {
        "word": input_data.word,
        "description": word_data.description,
        "suggestions": word_data.suggestions,
        "is_valid": word_data.is_valid,
        "examples": word_data.examples,
        "warning": word_data.warning,
        "type": word_data.type,
        "url": image_url,
        "image_path": image_path
    }
    
    stored_id = firebase_crud.insert_row("vocabulary_items", db_data)

    # Construct final response dictionary matching user requirements
    final_response = {
        "word": input_data.word,
        "description": word_data.description,
        "suggestions": word_data.suggestions,
        "is_valid": word_data.is_valid,
        "examples": word_data.examples,
        "warning": word_data.warning,
        "type": word_data.type,
        "URL": image_url, # Capitalized URL
        "image_path": image_path,
        "id": stored_id
    }

    return final_response

async def process_word_async(
    input_data: WordAgentInput,
    gemini_wrapper: GeminiWrapper
):
    result = await asyncio.to_thread(
        process_word_sync,
        input_data=input_data,
        gemini_wrapper=gemini_wrapper
    )
    if "error" in result:
        raise Exception(result["error"])
    return result

@app.get("/")
async def generate_word_content(
    words: Optional[str] = None,
    lang: Optional[str] = None,
    output_lang: Optional[str] = None,
    age: Optional[str] = None,
    level: Optional[str] = None,
    style: Optional[str] = None,
    country: Optional[str] = None,
    userid: Optional[str] = None,
    card_id: Optional[str] = None
):
    if not words:
        return {"error": "No words provided"}

    word_list = [w.strip() for w in words.split(",") if w.strip()]
    
    if not word_list:
        return {"error": "No valid words provided"}

    tasks = []
    for word in word_list:
        input_data = WordAgentInput(
            word=word,
            target_language=lang or "as the word",
            output_language=output_lang or lang,
            user_age=age or "None",
            proficiency_level=level or "None",
            image_style=style or "None",
            country=country or "None",
            userid=userid or "None",
            card_id=card_id or "None"
        )
        tasks.append(
            process_word_async(
                input_data=input_data,
                gemini_wrapper=gemini_wrapper
            )
        )
    
    results = await asyncio.gather(*tasks)
    
    return results

@app.get("/models")
async def list_available_models():
    """Debug endpoint to list all available models on this environment."""
    try:
        models = []
        for m in gemini_client.models.list():
            # filter for image models to keep it short
            name = m.name.lower() if m.name else ""
            display = m.display_name.lower() if m.display_name else ""
            if "image" in name or "image" in display or "imagen" in name:
                models.append({
                    "name": m.name,
                    "display_name": m.display_name
                })
        return {"environment": "Vertex AI" if gemini_client._api_client.vertexai else "AI Studio", "image_models": models}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
