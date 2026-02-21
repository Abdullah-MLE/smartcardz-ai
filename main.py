from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
import asyncio
from google import genai
from libs.GeminiWrapper.GeminiWrapper import GeminiWrapper
from libs.GeminiWrapper.models import InputParams, TextParams, ImageParams
from libs.SupabaseCRUD.SupabaseCRUD import SupabaseCRUD
from models import WordAgentResponse, WordAgentInput
from prompts import get_word_agent_system_prompt, get_word_agent_user_prompt

load_dotenv()

app = FastAPI()

def init_gemini_client():
    """Initialize and return a Gemini client."""
    client = genai.Client()
    return client

# Global Gemini Wrapper instance
gemini_client = init_gemini_client()
gemini_wrapper = GeminiWrapper(gemini_client)

# Initialize SupabaseCRUD
supabase_crud = SupabaseCRUD()

# -----------------------------------------------------------------------------
# Image Generation Helper
# -----------------------------------------------------------------------------
def generate_post_image(image_prompt: str, gemini_wrapper: GeminiWrapper, supabase_crud: SupabaseCRUD) -> Optional[str]:
    """Generates an image from a prompt and uploads it to Supabase."""
    if not image_prompt:
        return None

    input_params = InputParams(
        prompt=image_prompt,
        model="gemini-2.5-flash-image"
    )
    image_params = ImageParams(output_image_aspect_ratio="1:1") # Defaulting to 1:1 as per common square posts
    
    result = gemini_wrapper.generate_image(input_params=input_params, image_params=image_params)
    
    if not result["success"]:
        return None
    
    image_bytes = result["content"]
    
    # Upload to Supabase
    public_url = supabase_crud.upload_image(image_bytes, bucket_name="pics")
    return public_url

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
    if word_data.is_valid and word_data.image_prompt:
        image_url = generate_post_image(word_data.image_prompt, gemini_wrapper, supabase_crud)
        
    # 3. Store in Supabase
    db_data = {
        "word": input_data.word,
        "description": word_data.description,
        "suggestions": word_data.suggestions,
        "is_valid": word_data.is_valid,
        "examples": word_data.examples,
        "warning": word_data.warning,
        "type": word_data.type,
        "url": image_url
    }
    
    stored_id = None
    stored_item = supabase_crud.insert_row("vocabulary_items", db_data)
    if stored_item:
        stored_id = stored_item.get('id')

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
    country: Optional[str] = None
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
            country=country or "None"
        )
        tasks.append(
            process_word_async(
                input_data=input_data,
                gemini_wrapper=gemini_wrapper
            )
        )
    
    results = await asyncio.gather(*tasks)
    
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
