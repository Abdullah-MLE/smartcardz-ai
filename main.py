from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
import os
import random
import asyncio
from datetime import datetime
from supabase import create_client, Client
from google import genai
from GeminiWrapper import GeminiWrapper
from models import InputParams, TextParams, ImageParams, WordAgentResponse
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

# -----------------------------------------------------------------------------
# Supabase CRUD
# -----------------------------------------------------------------------------
class SupabaseCRUD:
    def __init__(self):
        self.supabase_client = self._init_supabase_client()

    def _init_supabase_client(self) -> Client:
        url = os.environ.get("SUPABASE_URL") 
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
             print("Warning: SUPABASE_URL or SUPABASE_KEY not found in environment variables.")
             # You might want to handle this more gracefully or let it fail later
             
        return create_client(url, key)

    def _generate_unique_filename(self, extension="png") -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"img_{timestamp}_{random.randint(0, 1000000)}.{extension}"

    def _get_public_url(self, bucket_name: str, file_name: str) -> str:
        return self.supabase_client.storage.from_(bucket_name).get_public_url(file_name)

    def upload_image(
        self,
        image_bytes: bytes,
        file_name: str = None,
        bucket_name="pics",
        content_type="image/png") -> str:

        if not file_name:
            file_name = self._generate_unique_filename()

        self.supabase_client.storage.from_(bucket_name).upload(
            path=file_name,
            file=image_bytes,
            file_options={
                "content-type": content_type,
                "upsert": "true"
            }
        )

        return self._get_public_url(bucket_name, file_name)

    def insert_vocabulary_item(self, data: dict):
        response = self.supabase_client.table("vocabulary_items").insert(data).execute()
        return response.data[0] if response.data else None

# Initialize SupabaseCRUD
supabase_crud = SupabaseCRUD()

# -----------------------------------------------------------------------------
# Image Generation Helper
# -----------------------------------------------------------------------------
def generate_post_image(image_prompt: str, gemini_wrapper: GeminiWrapper, supabase_crud: SupabaseCRUD) -> Optional[str]:
    """Generates an image from a prompt and uploads it to Supabase."""
    if not image_prompt:
        return None

    print(f"Generating image for prompt: {image_prompt[:50]}...")
    
    input_params = InputParams(prompt=image_prompt)
    image_params = ImageParams(output_image_aspect_ratio="1:1") # Defaulting to 1:1 as per common square posts
    
    result = gemini_wrapper.generate_image(input_params=input_params, image_params=image_params)
    
    if not result["success"]:
        print(f"Image generation failed: {result.get('error')}")
        return None
    
    image_bytes = result["content"]
    # Upload to Supabase
    try:
        public_url = supabase_crud.upload_image(image_bytes, bucket_name="pics")
        print(f"Image uploaded successfully: {public_url}")
        return public_url
    except Exception as e:
        print(f"Image upload failed: {e}")
        return None

# -----------------------------------------------------------------------------
# Main Logic
# -----------------------------------------------------------------------------
def process_word_sync(
    word: str,
    user_age: str,
    proficiency_level: str,
    image_style: str,
    country: str,
    target_language: str,
    gemini_wrapper: GeminiWrapper
):
    # 1. Generate Text Content
    system_prompt = get_word_agent_system_prompt()
    user_prompt = get_word_agent_user_prompt(
        word=word,
        target_language=target_language,
        user_age=user_age,
        proficiency_level=proficiency_level,
        image_style=image_style,
        country=country
    )
    
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
        "word": word,
        "description": word_data.description,
        "suggestions": word_data.suggestions,
        "is_valid": word_data.is_valid,
        "examples": word_data.examples,
        "warning": word_data.warning,
        "type": word_data.type,
        "url": image_url
    }
    
    stored_id = None
    try:
            stored_item = supabase_crud.insert_vocabulary_item(db_data)
            if stored_item:
                stored_id = stored_item.get('id')
            print(f"Stored vocabulary item: {word}")
    except Exception as e:
            # Just print error, don't fail the whole response if DB insert fails
            print(f"Failed to store vocabulary item: {e}")
            # We might want to raise here if we want the retry logic to catch DB failures too
            raise e

    # Construct final response dictionary matching user requirements
    final_response = {
        "word": word,
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
    word: str,
    user_age: str,
    proficiency_level: str,
    image_style: str,
    country: str,
    target_language: str,
    gemini_wrapper: GeminiWrapper
):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"Processing word: {word} (Attempt {attempt + 1}/{max_retries})")
            result = await asyncio.to_thread(
                process_word_sync,
                word=word,
                user_age=user_age,
                proficiency_level=proficiency_level,
                image_style=image_style,
                country=country,
                target_language=target_language,
                gemini_wrapper=gemini_wrapper
            )
            if "error" in result:
                raise Exception(result["error"])
            return result
        except Exception as e:
            print(f"Error processing {word} (Attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return {"word": word, "error": f"Failed after {max_retries} attempts. Last error: {str(e)}"}
            await asyncio.sleep(1) # Simple backoff

@app.get("/")
async def generate_word_content(
    words: Optional[str] = None,
    lang: Optional[str] = None,
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
        tasks.append(
            process_word_async(
                word=word,
                target_language=lang,
                user_age=age,
                proficiency_level=level,
                image_style=style,
                country=country,
                gemini_wrapper=gemini_wrapper
            )
        )
    
    results = await asyncio.gather(*tasks)
    
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
