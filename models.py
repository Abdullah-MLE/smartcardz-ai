"""
Pydantic Models for WordAgent API
==================================
Organized parameter models.
"""

from typing import Optional
from pydantic import BaseModel


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
    description: Optional[str] = None
    is_valid: bool
    type: Optional[str] = None
    examples: Optional[str] = None
    image_prompt: Optional[str] = None
    image_url: Optional[str] = None
    warning: Optional[str] = None
    suggestions: Optional[str] = None
