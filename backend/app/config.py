"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    """Typed settings for the Lyftr AI Chatbot backend."""

    model_config = SettingsConfigDict(
        env_file=(str(_ROOT_ENV), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = Field(default="Lyftr AI Chatbot", alias="APP_NAME")
    database_url: str = Field(default="sqlite:///./data/app.db", alias="DATABASE_URL")

    llm_api_keys: str = Field(default="", alias="LLM_API_KEYS")
    llm_model: str = Field(default="gemini/gemini-2.0-flash", alias="LLM_MODEL")
    llm_models: str = Field(default="", alias="LLM_MODELS")
    llm_base_url: str = Field(default="", alias="LLM_BASE_URL")
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=1024, ge=1, le=16384, alias="LLM_MAX_TOKENS")
    llm_timeout_seconds: float = Field(default=30.0, gt=0.0, alias="LLM_TIMEOUT_SECONDS")

    cache_enabled: bool = Field(default=True, alias="CACHE_ENABLED")
    enable_failure_simulation: bool = Field(
        default=False, alias="ENABLE_FAILURE_SIMULATION"
    )

    default_system_prompt: str = Field(
        default=(
            "You are Lyftr AI, a helpful, concise, and knowledgeable assistant. "
            "Provide clear, accurate answers."
        ),
        alias="DEFAULT_SYSTEM_PROMPT",
    )
    max_history_messages: int = Field(default=40, ge=1, le=200, alias="MAX_HISTORY_MESSAGES")

    allowed_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173",
        alias="ALLOWED_ORIGINS",
    )

    @property
    def parsed_api_keys(self) -> list[str]:
        """Parse comma-separated LLM_API_KEYS into an ordered list."""
        placeholders = {
            "",
            "your-first-api-key",
            "your-second-api-key",
            "changeme",
            "placeholder",
            "replace-me",
        }
        return [
            key.strip()
            for key in self.llm_api_keys.split(",")
            if key.strip().lower() not in placeholders
            and not key.strip().lower().startswith("your-")
        ]

    @property
    def parsed_models(self) -> list[str]:
        """Parse optional comma-separated LLM_MODELS into a list."""
        return [model.strip() for model in self.llm_models.split(",") if model.strip()]

    @property
    def cors_origins(self) -> list[str]:
        """Parse comma-separated ALLOWED_ORIGINS into a list."""
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
