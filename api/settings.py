import os
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    # Core configuration
    db_dsn: str = Field(
        default="postgresql://cbwinslow:123qweasd@localhost:5432/cbw_rag",
        env="CBW_RAG_DATABASE",
    )
    ollama_url: str = Field(default="http://localhost:11434", env="CBW_RAG_OLLAMA_URL")
    embedding_model: str = Field(default="nomic-embed-text", env="CBW_RAG_EMBEDDING_MODEL")
    llm_model: str = Field(default="llama3.1:8b", env="CBW_RAG_LLM_MODEL")
    # Domain for external exposure (e.g., cloudcurio.cc)
    domain: str = Field(default="cloudcurio.cc", env="CLOUDCURIO_DOMAIN")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
