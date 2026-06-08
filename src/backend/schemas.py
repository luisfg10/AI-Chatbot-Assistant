"""Pydantic schemas for request and response models in the backend API."""
from pydantic import BaseModel


class AvailableModelsResponse(BaseModel):
    """Schema for the response containing available LLM models."""
    models: list[str]
    default_model: str


class AvailablePersonalitiesResponse(BaseModel):
    """Schema for the response containing supported chatbot personalities."""
    personalities: list[str]
    default_personality: str


class ChatRequest(BaseModel):
    """Schema for incoming chat messages from the frontend."""
    message: str


class ChatResponse(BaseModel):
    """Schema for outgoing chat responses to the frontend."""
    response: str
