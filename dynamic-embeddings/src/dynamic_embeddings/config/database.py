"""Database configuration for PGVector connections."""

from typing import Optional

from pydantic import BaseModel, Field, validator


class DatabaseConfig(BaseModel):
    """Configuration for PostgreSQL database with PGVector extension."""

    host: str = Field(description="PostgreSQL server hostname")
    port: int = Field(default=5432, description="PostgreSQL server port")
    database: str = Field(description="Database name")
    username: str = Field(description="Database username")
    password: str = Field(description="Database password")
    ssl_mode: str = Field(default="prefer", description="SSL connection mode")
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=20, description="Maximum pool overflow")
    pool_timeout: int = Field(default=30, description="Pool timeout in seconds")
    pool_recycle: int = Field(default=3600, description="Pool recycle time in seconds")

    @validator("port")
    def validate_port(cls, v: int) -> int:
        """Validate port number range."""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v

    @validator("ssl_mode")
    def validate_ssl_mode(cls, v: str) -> str:
        """Validate SSL mode."""
        valid_modes = ["disable", "allow", "prefer", "require", "verify-ca", "verify-full"]
        if v not in valid_modes:
            raise ValueError(f"SSL mode must be one of: {', '.join(valid_modes)}")
        return v

    @property
    def connection_url(self) -> str:
        """Get SQLAlchemy connection URL."""
        password_part = f":{self.password}" if self.password else ""
        return f"postgresql://{self.username}{password_part}@{self.host}:{self.port}/{self.database}"

    @property
    def async_connection_url(self) -> str:
        """Get async SQLAlchemy connection URL."""
        password_part = f":{self.password}" if self.password else ""
        return f"postgresql+asyncpg://{self.username}{password_part}@{self.host}:{self.port}/{self.database}"

    def to_dict(self) -> dict:
        """Convert to dictionary for psycopg connection."""
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.database,
            "user": self.username,
            "password": self.password,
            "sslmode": self.ssl_mode,
        }