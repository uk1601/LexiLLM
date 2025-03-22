"""
Schema definitions for LexiLLM
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field

class IntentClassifier(BaseModel):
    """Schema for classifying the user's intent."""
    intent: Literal["LLM_FUNDAMENTALS", "LLM_IMPLEMENTATION", "LLM_COMPARISON", "LLM_NEWS", "UNKNOWN"] = Field(
        description="The detected intent from the user's query"
    )
    confidence: float = Field(
        description="Confidence level for the intent classification (0-1)",
        ge=0.0,
        le=1.0
    )

class UserInfo(BaseModel):
    """Schema for extracting user information."""
    name: Optional[str] = Field(
        None, description="The user's name if mentioned"
    )
    technical_level: Optional[str] = Field(
        None, description="User's technical expertise with LLMs (beginner, intermediate, advanced)"
    )
    project_stage: Optional[str] = Field(
        None, description="User's project stage (planning, development, optimization)"
    )
    comparison_criterion: Optional[str] = Field(
        None, description="User's model comparison criterion (accuracy, speed, cost)"
    )
    interest_area: Optional[str] = Field(
        None, description="User's area of interest in LLM developments (research, applications)"
    )
