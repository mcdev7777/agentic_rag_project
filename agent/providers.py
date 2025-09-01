
import os
from typing import Optional
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIChatModel
import openai
from dotenv import load_dotenv


load_dotenv()


def get_llm_model(model_choice: Optional[str] = None) -> OpenAIChatModel:
    """
    Get LLM model configuration based on environment variables.
    
    Args:
        model_choice: Optional override for model choice
    
    Returns:
        Configured OpenAI-compatible model
    """
    llm_choice = model_choice or os.getenv('LLM_CHOICE', 'gpt-4-turbo-preview')
    api_key = os.getenv('OPENAI_API_KEY', 'LLM_API_KEY')

    provider = OpenAIProvider(api_key=api_key)
    return OpenAIChatModel(llm_choice, provider=provider)


def get_embedding_client() -> openai.AsyncOpenAI:
    """
    Get embedding client configuration based on environment variables.
    
    Returns:
        Configured OpenAI-compatible client for embeddings
    """
    api_key = os.getenv('OPENAI_API_KEY')
    
    return openai.AsyncOpenAI(
        api_key=api_key
    )


def get_embedding_model() -> str:
    """
    Get embedding model name from environment.
    
    Returns:
        Embedding model name
    """
    return os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')

