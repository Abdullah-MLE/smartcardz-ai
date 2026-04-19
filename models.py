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
    description: Optional[str] = Field(None, description="Explanation or definition of the word")
    is_valid: bool = Field(..., description="True if the word is valid, False otherwise")
    type: Optional[str] = Field(None, description="Grammatical part of speech")
    examples: Optional[str] = Field(None, description="Exactly two real-life sentences separated by semicolons")
    image_prompt: Optional[str] = Field(None, description="Rich and detailed living scene description used for image generation")
    image_url: Optional[str] = None
    warning: Optional[str] = Field(None, description="Warning message if the word is invalid")
    suggestions: Optional[str] = Field(None, description="Suggestions if the word is invalid")
