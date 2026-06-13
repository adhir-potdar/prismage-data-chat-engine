"""Database connection management for PGVector."""

import os
import logging
from typing import Optional, Dict, Any
from sqlalchemy import create_engine, Engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
import psycopg
from contextlib import contextmanager


class DatabaseConnection:
    """Manages database connections for PGVector embeddings storage."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database_url: Optional[str] = None,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        echo: bool = False
    ):
        """Initialize database connection.

        Args:
            host: PostgreSQL host
            port: PostgreSQL port
            database: Database name
            username: Username
            password: Password
            database_url: Complete connection URL (overrides individual params)
            pool_size: Connection pool size
            max_overflow: Maximum overflow connections
            pool_timeout: Pool timeout in seconds
            echo: Whether to echo SQL statements
        """
        self.logger = logging.getLogger(__name__)

        # Build connection URL
        if database_url:
            self.database_url = database_url
        else:
            # Use environment variables as fallbacks
            host = host or os.getenv('POSTGRES_HOST', 'localhost')
            port = port or int(os.getenv('POSTGRES_PORT', '5432'))
            database = database or os.getenv('POSTGRES_DB', 'vectordb')
            username = username or os.getenv('POSTGRES_USER', 'postgres')
            password = password or os.getenv('POSTGRES_PASSWORD', 'omkar')

            self.database_url = f"postgresql+psycopg://{username}:{password}@{host}:{port}/{database}"

        # Create engine with connection pooling
        self.engine = create_engine(
            self.database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            echo=echo,
            # Important for pgvector: ensure connections support the extension
            connect_args={
                "application_name": "dynamic_embeddings",
                "options": "-c timezone=UTC"
            }
        )

        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

        # Flag to suppress error logging for expected failures
        self._suppress_error_logging = False

        self.logger.info(f"Database connection initialized: {self._safe_url()}")

    def _safe_url(self) -> str:
        """Return connection URL without password for logging."""
        if '@' in self.database_url:
            parts = self.database_url.split('@')
            if ':' in parts[0]:
                user_parts = parts[0].split(':')
                safe_user = user_parts[0] + ':***'
                return safe_user + '@' + parts[1]
        return self.database_url.replace(':***', ':***')

    def test_connection(self) -> Dict[str, Any]:
        """Test database connection and return connection info.

        Returns:
            Dictionary with connection test results
        """
        try:
            with self.engine.connect() as conn:
                # Test basic connection
                result = conn.execute(text("SELECT 1 as test")).scalar()

                # Check PostgreSQL version
                pg_version = conn.execute(text("SELECT version()")).scalar()

                # Check if pgvector extension is available
                vector_available = conn.execute(text(
                    "SELECT EXISTS(SELECT 1 FROM pg_available_extensions WHERE name = 'vector')"
                )).scalar()

                # Check if pgvector extension is installed
                vector_installed = conn.execute(text(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                )).scalar()

                return {
                    'connected': True,
                    'test_query_result': result,
                    'postgresql_version': pg_version.split()[1] if pg_version else 'unknown',
                    'pgvector_available': vector_available,
                    'pgvector_installed': vector_installed,
                    'database_url': self._safe_url()
                }

        except Exception as e:
            self.logger.error(f"Database connection test failed: {e}")
            return {
                'connected': False,
                'error': str(e),
                'database_url': self._safe_url()
            }

    @contextmanager
    def get_session(self):
        """Get a database session with automatic cleanup.

        Yields:
            SQLAlchemy session

        Example:
            with db.get_session() as session:
                results = session.query(EmbeddingRecord).all()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            # Only log error if not suppressed (for expected failures like checking table existence)
            if not self._suppress_error_logging:
                self.logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    @contextmanager
    def suppress_errors(self):
        """Context manager to suppress error logging for expected failures.

        Usage:
            with db.suppress_errors():
                # Query that might fail expectedly (e.g., checking if table exists)
                session.query(Model).limit(1).first()
        """
        original_value = self._suppress_error_logging
        self._suppress_error_logging = True
        try:
            yield
        finally:
            self._suppress_error_logging = original_value

    def get_raw_connection(self):
        """Get a raw database connection for advanced operations.

        Returns:
            SQLAlchemy connection
        """
        return self.engine.connect()

    def close(self) -> None:
        """Close all database connections."""
        try:
            self.engine.dispose()
            self.logger.info("Database connections closed")
        except Exception as e:
            self.logger.error(f"Error closing database connections: {e}")

    def get_connection_info(self) -> Dict[str, Any]:
        """Get information about the database connection.

        Returns:
            Connection configuration info
        """
        return {
            'database_url': self._safe_url(),
            'pool_size': self.engine.pool.size(),
            'checked_in_connections': self.engine.pool.checkedin(),
            'checked_out_connections': self.engine.pool.checkedout(),
            'overflow_connections': self.engine.pool.overflow(),
            'invalid_connections': getattr(self.engine.pool, 'invalidated', lambda: 0)(),
        }

    def execute_raw_sql(self, sql: str, params: Optional[Dict] = None) -> Any:
        """Execute raw SQL statement.

        Args:
            sql: SQL statement
            params: Query parameters

        Returns:
            Query result
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql), params or {})
                conn.commit()
                return result

        except Exception as e:
            self.logger.error(f"Raw SQL execution failed: {e}")
            raise

