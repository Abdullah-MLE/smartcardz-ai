"""
Pydantic Models for WordAgent API
==================================
Organized parameter models.
"""

from typing import Optional
from pydantic import BaseModel, Field


class WordAgentInput(BaseModel):
    """Schema for Word Agent Input parameters."""
    word: str
    target_language: str = "as the word"
    output_language: str = "Arabic"
    user_age: str = "None"
    country: str = "None"
    image_style: str = "None"
    proficiency_level: str = "None"


class WordAgentResponse(BaseModel):
    """Schema for Word Agent response."""
    description: str = Field(..., description="Explanation or definition of the word. Empty string if invalid.")
    is_valid: bool = Field(..., description="True if the word is valid, False otherwise")
    type: str = Field(..., description="Grammatical part of speech. Empty string if invalid.")
    examples: str = Field(..., description="Exactly two real-life sentences separated by semicolons. Empty string if invalid.")
    image_prompt: str = Field(..., description="Rich and detailed living scene description used for image generation. Empty string if invalid.")
    image_url: Optional[str] = None
    warning: str = Field(..., description="Warning message if the word is invalid. Empty string if valid.")
    suggestions: str = Field(..., description="Suggestions if the word is invalid. Empty string if valid.")
