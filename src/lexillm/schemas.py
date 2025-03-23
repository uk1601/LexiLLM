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
    reasoning: str = Field(
        description="Explanation of why this intent was selected", 
        default=""
    )
    topics: list = Field(
        description="List of relevant LLM topics in the query",
        default_factory=list
    )

class DomainRelevance(BaseModel):
    """Schema for determining if a query is related to LLMs."""
    is_relevant: bool = Field(
        description="Whether the query is related to LLMs or closely associated AI technologies"
    )
    confidence: float = Field(
        description="Confidence level for the relevance assessment (0-1)",
        ge=0.0,
        le=1.0
    )
    related_topics: list = Field(
        description="List of LLM topics the query relates to",
        default_factory=list
    )
    reasoning: str = Field(
        description="Explanation of the classification"
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
