"""
Gemini Configuration
====================
Simple centralized config using Pydantic BaseModel.
"""

from pydantic import BaseModel
from typing import List


class GeminiConfig(BaseModel):
    """Gemini wrapper configuration - all settings in one place."""
    
    # Text Generation Models
    default_text_model: str = "gemini-3-flash-preview"
    text_fallback_models: List[str] = [
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
    ]
    
    # Image Generation Models
    default_image_model: str = "gemini-3-pro-image-preview"
    image_models: List[str] = [
        "gemini-3-pro-image-preview",
        "gemini-2.5-flash-image",
    ]
    
    # Image Output Settings
    default_output_image_size: str = "1K"
    default_output_image_aspect_ratio: str = "1:1"
    
    # Image Processing (input resizing)
    default_image_max_dimension: int = 300
    default_media_resolution: str = "MEDIA_RESOLUTION_UNSPECIFIED"
    
    # Retry Settings
    retry_delay_seconds: float = 2.0
    
    # Text Generation Retry
    text_max_retries_per_model: int = 2
    text_overall_max_retries: int = 10
    
    # Image Generation Retry
    image_max_retries_per_model: int = 2
    image_overall_max_retries: int = 5
    
    # Default Instructions
    default_system_instruction: str = "You are a helpful assistant"
    default_response_mime_type: str = "application/json"


# Global config instance
config = GeminiConfig()
