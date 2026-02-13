from pydantic_settings import BaseSettings
from pydantic import Field
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # OpenRouter API
    OPENROUTER_API_KEY: str = Field(default="")
    OPENROUTER_BASE_URL: str = Field(default="https://openrouter.ai/api/v1")
    OPENROUTER_MODEL_MAIN: str = Field(default="meta-llama/llama-3.3-70b-instruct:free")
    OPENROUTER_MODEL_FAST: str = Field(default="meta-llama/llama-3.2-3b-instruct:free")
    OPENROUTER_MODEL_VISION: str = Field(default="qwen/qwen2-vl-72b-instruct:free")
    
    # Database - PostgreSQL
    DATABASE_URL: str = Field(default="postgresql://rag_user:rag_pass_2026@localhost:5432/researchforge")
    POSTGRES_USER: str = Field(default="rag_user")
    POSTGRES_PASSWORD: str = Field(default="rag_pass_2026")
    POSTGRES_DB: str = Field(default="researchforge")
    
    # ChromaDB
    CHROMA_HOST: str = Field(default="localhost")
    CHROMA_PORT: int = Field(default=8000)
    CHROMA_COLLECTION: str = Field(default="researchforge_docs")
    
    # Local Models
    EMBEDDING_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    RERANKER_MODEL: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # Application Settings
    ENVIRONMENT: str = Field(default="development")
    LOG_LEVEL: str = Field(default="INFO")
    CHUNK_SIZE: int = Field(default=512)
    CHUNK_OVERLAP: int = Field(default=50)
    TOP_K_RETRIEVAL: int = Field(default=5)
    
    # LangSmith Tracing
    LANGSMITH_TRACING: bool = Field(default=True)
    LANGSMITH_ENDPOINT: str = Field(default="https://api.smith.langchain.com")
    LANGSMITH_API_KEY: str = Field(default="")
    LANGSMITH_PROJECT: str = Field(default="ResearchForge")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        # Allow extra fields from .env without validation errors
        extra = "ignore"

# Create settings instance
settings = Settings()
