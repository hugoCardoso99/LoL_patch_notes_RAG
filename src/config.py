"""Centralized configuration loaded from environment variables."""

import os

from dotenv import load_dotenv
from pydantic import BaseModel, computed_field

load_dotenv()


class DatabaseConfig(BaseModel):
    host: str = os.getenv("POSTGRES_HOST", "127.0.0.1")
    port: int = int(os.getenv("POSTGRES_PORT", "5555"))
    user: str = os.getenv("POSTGRES_USER", "raguser")
    password: str = os.getenv("POSTGRES_PASSWORD", "ragpass")
    database: str = os.getenv("POSTGRES_DB", "lol_rag")

    @computed_field
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class ModelConfig(BaseModel):
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    llm_model: str = os.getenv("LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "512"))

class AppConfig(BaseModel):
    db: DatabaseConfig = DatabaseConfig()
    model: ModelConfig = ModelConfig()


config = AppConfig()
