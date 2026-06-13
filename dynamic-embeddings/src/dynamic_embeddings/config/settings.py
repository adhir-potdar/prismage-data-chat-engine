"""Application settings and configuration management."""

import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global application settings."""

    # Database Configuration
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_db: str = Field(default="vectordb", env="POSTGRES_DB")
    postgres_user: str = Field(default="postgres", env="POSTGRES_USER")
    postgres_password: str = Field(default="", env="POSTGRES_PASSWORD")
    postgres_ssl_mode: str = Field(default="prefer", env="POSTGRES_SSL_MODE")

    # OpenAI Configuration
    openai_api_key: str = Field(env="OPENAI_API_KEY")
    openai_model: str = Field(default="text-embedding-3-large", env="OPENAI_EMBEDDING_MODEL")
    openai_max_retries: int = Field(default=3, env="OPENAI_MAX_RETRIES")
    openai_timeout: int = Field(default=60, env="OPENAI_TIMEOUT")

    # Engine Configuration
    default_collection: str = Field(default="documents", env="DEFAULT_COLLECTION")
    chunk_size_limit_mb: float = Field(default=10.0, env="CHUNK_SIZE_LIMIT_MB")
    enable_streaming: bool = Field(default=True, env="ENABLE_STREAMING")
    batch_size: int = Field(default=100, env="BATCH_SIZE")
    max_workers: int = Field(default=4, env="MAX_WORKERS")

    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    enable_json_logs: bool = Field(default=False, env="ENABLE_JSON_LOGS")

    # Strategy Configuration
    enable_auto_strategy: bool = Field(default=True, env="ENABLE_AUTO_STRATEGY")
    default_strategy: str = Field(default="hierarchical_chunking", env="DEFAULT_STRATEGY")
    strategy_confidence_threshold: float = Field(
        default=0.7, env="STRATEGY_CONFIDENCE_THRESHOLD"
    )

    # Performance Configuration
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def database_url(self) -> str:
        """Get PostgreSQL connection URL."""
        password_part = f":{self.postgres_password}" if self.postgres_password else ""
        return (
            f"postgresql://{self.postgres_user}{password_part}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return os.getenv("ENV", "development").lower() == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return os.getenv("ENV", "development").lower() == "production"


# Global settings instance
settings = Settings()