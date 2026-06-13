"""Database schema for PGVector embeddings storage."""

import warnings
from sqlalchemy import (
    Column, Integer, String, Text, REAL, TIMESTAMP, BOOLEAN, JSON,
    Index, UniqueConstraint, create_engine, text, Table, MetaData
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
import logging
import threading
import re
from typing import Optional, Dict, Type, Any

# Suppress SQLAlchemy warning about duplicate model registration
# This is expected behavior when using dynamic namespace-based models with caching
warnings.filterwarnings('ignore', message='.*declarative base already contains a class.*', category=Warning)

Base = declarative_base()

# Global cache for collection metadata models
_collection_metadata_models = {}


class EmbeddingRecord(Base):
    """SQLAlchemy model for embedding storage in PGVector."""

    __tablename__ = 'embeddings'

    # Primary Key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Vector Data
    embedding = Column(Vector(1536))  # OpenAI text-embedding-3-large dimension (configured)
    embedding_model = Column(String(100), nullable=False)
    embedding_created_at = Column(TIMESTAMP(timezone=True), nullable=False)

    # Content Identity
    chunk_id = Column(String(4096), unique=True, nullable=False)
    text = Column(Text, nullable=False)
    text_hash = Column(String(64), nullable=False)
    text_length = Column(Integer)

    # Hierarchical Context
    path = Column(Text)
    level = Column(Integer)
    parent_id = Column(String(4096))
    children_ids = Column(JSONB)

    # Source Tracking
    source_file = Column(Text)
    document_id = Column(String(4096))
    collection_name = Column(String(100))

    # Content Classification
    content_type = Column(String(50))
    value_types = Column(JSONB)
    key_count = Column(Integer)

    # Strategy & Quality
    strategy = Column(String(50))
    confidence = Column(REAL)
    semantic_density = Column(REAL)

    # Domain & Analysis
    domain_type = Column(String(100))
    entity_types = Column(JSONB)
    performance_metrics = Column(JSONB)
    dimension_value = Column(String(1024), index=True)  # Extracted from dimension_analyses keys (APP, AMP, etc.)
    reasoning_content = Column(JSONB)

    # Technical Metadata
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    version = Column(String(20), default='1.0')
    processing_pipeline = Column(String(50), default='vector_embeddings')

    __table_args__ = (
        # Unique constraints
        UniqueConstraint('chunk_id', name='uq_embeddings_chunk_id'),
        UniqueConstraint('text_hash', 'document_id', name='uq_embeddings_text_hash_document'),

        # Vector similarity index (created separately due to pgvector requirements)
        # CREATE INDEX embeddings_vector_idx ON embeddings USING ivfflat (embedding vector_cosine_ops)

        # Search indexes
        Index('ix_embeddings_source', 'source_file', 'document_id'),
        Index('ix_embeddings_strategy', 'strategy', 'confidence'),
        Index('ix_embeddings_content_type', 'content_type', 'domain_type'),
        Index('ix_embeddings_path', 'path'),
        Index('ix_embeddings_collection', 'collection_name'),
        Index('ix_embeddings_level', 'level'),
        Index('ix_embeddings_dimension_value', 'dimension_value'),
        Index('ix_embeddings_created', 'created_at'),
    )

    def __repr__(self):
        return f"<EmbeddingRecord(id={self.id}, chunk_id='{self.chunk_id}', strategy='{self.strategy}')>"


def get_collection_metadata_model(namespace: str = 'default'):
    """
    Factory function to create namespace-specific collection metadata model.

    Table name format: embeddings_collection_metadata_<namespace>
    Examples: embeddings_collection_metadata_default, embeddings_collection_metadata_revenue_mgmt

    Args:
        namespace: Namespace identifier (lowercase alphanumeric + underscore)

    Returns:
        SQLAlchemy model class for the namespace-specific collection metadata table
    """
    # Check cache first to avoid re-creating the same class
    global _collection_metadata_models
    if namespace in _collection_metadata_models:
        return _collection_metadata_models[namespace]

    # Validate namespace
    if not re.match(r'^[a-z0-9_]+$', namespace):
        raise ValueError(f"Invalid namespace: {namespace}. Must contain only lowercase letters, numbers, and underscores.")

    table_name = f'embeddings_collection_metadata_{namespace}'

    # Create dynamic model class with unique name from the start
    class_name = f'CollectionMetadata_{namespace}'

    # Create dynamic model class using type() to set name immediately
    CollectionMetadata = type(
        class_name,
        (Base,),
        {
            '__tablename__': table_name,
            '__table_args__': {'extend_existing': True},

            # Primary key
            'collection_name': Column(String(200), primary_key=True),

            # Parsed components from collection name
            'dimension': Column(String(100), nullable=False, index=True),
            'time_granularity': Column(String(10), nullable=False, index=True),
            'dimension_values': Column(JSONB),  # Array of dimension values in this collection (e.g., ['APP', 'AMP', 'DESK'])

            # Date ranges (YYYYMMDD format as integers for efficient range queries)
            'period1_start_date': Column(Integer, nullable=False, index=True),
            'period1_end_date': Column(Integer, nullable=False, index=True),
            'period2_start_date': Column(Integer, nullable=False, index=True),
            'period2_end_date': Column(Integer, nullable=False, index=True),

            # Statistics (optional, for display)
            'total_embeddings': Column(Integer, default=0),
            'last_updated_at': Column(TIMESTAMP(timezone=True), nullable=False),
            'created_at': Column(TIMESTAMP(timezone=True), server_default=func.now()),

            '__repr__': lambda self: f"<{class_name}(name='{self.collection_name}', dim='{self.dimension}', gran='{self.time_granularity}')>",
            '__doc__': """Metadata table for fast collection lookups.

        Stores parsed information from collection names to enable fast filtering
        by dimension, time granularity, and date ranges without expensive queries.

        Collection name format: {dimension}_{granularity}_{start1}_{end1}_vs_{start2}_{end2}
        Example: property_geo_device_qoq_20250601_20250831_vs_20250901_20251130

        User provides single date range which is matched against both period1 and period2.
        """
        }
    )

    # Cache the model to prevent re-creation
    _collection_metadata_models[namespace] = CollectionMetadata

    return CollectionMetadata


class NamespaceTableFactory:
    """Factory for dynamically creating namespace-specific embedding tables."""

    # Class-level constants
    MAX_NAMESPACE_LENGTH = 50
    NAMESPACE_PATTERN = re.compile(r'^[a-z0-9_]+$')
    RESERVED_NAMESPACES = {'backup', 'temp', 'tmp', 'system', 'admin'}

    def __init__(self, engine):
        """Initialize the namespace table factory.

        Args:
            engine: SQLAlchemy engine instance
        """
        self.engine = engine
        self.logger = logging.getLogger(__name__)
        self._namespace_models: Dict[str, Type[Base]] = {}
        self._lock = threading.Lock()

    def validate_namespace(self, namespace: str) -> None:
        """Validate namespace name.

        Args:
            namespace: Namespace name to validate

        Raises:
            ValueError: If namespace is invalid
        """
        if not namespace:
            raise ValueError("Namespace cannot be empty")

        # Normalize to lowercase
        namespace = namespace.lower()

        if len(namespace) > self.MAX_NAMESPACE_LENGTH:
            raise ValueError(f"Namespace too long (max {self.MAX_NAMESPACE_LENGTH} chars): {namespace}")

        if not self.NAMESPACE_PATTERN.match(namespace):
            raise ValueError(f"Invalid namespace name (use only lowercase letters, numbers, underscore): {namespace}")

        if namespace in self.RESERVED_NAMESPACES:
            raise ValueError(f"Reserved namespace name: {namespace}")

    def get_table_name(self, namespace: str) -> str:
        """Get table name for a namespace.

        Args:
            namespace: Namespace identifier

        Returns:
            Table name (e.g., 'embeddings_prod')
        """
        namespace = namespace.lower()
        return f"embeddings_{namespace}"

    def table_exists(self, namespace: str) -> bool:
        """Check if namespace table exists in database.

        Args:
            namespace: Namespace identifier

        Returns:
            True if table exists
        """
        table_name = self.get_table_name(namespace)
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.tables "
                    "WHERE table_name = :table_name)"
                ), {"table_name": table_name}).scalar()
                return bool(result)
        except Exception as e:
            self.logger.error(f"Failed to check if table {table_name} exists: {e}")
            return False

    def get_or_create_model(self, namespace: str) -> Type[Base]:
        """Get or create SQLAlchemy model for namespace.

        Args:
            namespace: Namespace identifier

        Returns:
            SQLAlchemy model class for the namespace table
        """
        namespace = namespace.lower()
        self.validate_namespace(namespace)

        # Check cache first
        if namespace in self._namespace_models:
            return self._namespace_models[namespace]

        # Thread-safe model creation
        with self._lock:
            # Double-check after acquiring lock
            if namespace in self._namespace_models:
                return self._namespace_models[namespace]

            # Create dynamic model
            table_name = self.get_table_name(namespace)

            # Define table columns (same as EmbeddingRecord)
            table = Table(
                table_name,
                Base.metadata,

                # Primary Key
                Column('id', Integer, primary_key=True, autoincrement=True),

                # Vector Data
                Column('embedding', Vector(1536)),
                Column('embedding_model', String(100), nullable=False),
                Column('embedding_created_at', TIMESTAMP(timezone=True), nullable=False),

                # Content Identity
                Column('chunk_id', String(4096), unique=True, nullable=False),
                Column('text', Text, nullable=False),
                Column('text_hash', String(64), nullable=False),
                Column('text_length', Integer),

                # Hierarchical Context
                Column('path', Text),
                Column('level', Integer),
                Column('parent_id', String(4096)),
                Column('children_ids', JSONB),

                # Source Tracking
                Column('source_file', Text),
                Column('document_id', String(4096)),
                Column('collection_name', String(100)),

                # Content Classification
                Column('content_type', String(50)),
                Column('value_types', JSONB),
                Column('key_count', Integer),

                # Strategy & Quality
                Column('strategy', String(50)),
                Column('confidence', REAL),
                Column('semantic_density', REAL),

                # Domain & Analysis
                Column('domain_type', String(100)),
                Column('entity_types', JSONB),
                Column('performance_metrics', JSONB),
                Column('dimension_value', String(1024), index=True),  # Extracted from dimension_analyses keys
                Column('reasoning_content', JSONB),

                # Technical Metadata
                Column('created_at', TIMESTAMP(timezone=True), server_default=func.now()),
                Column('updated_at', TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now()),
                Column('version', String(20), default='1.0'),
                Column('processing_pipeline', String(50), default='vector_embeddings'),

                # Constraints and Indexes
                UniqueConstraint('chunk_id', name=f'uq_{table_name}_chunk_id'),
                UniqueConstraint('text_hash', 'document_id', name=f'uq_{table_name}_text_hash_document'),
                Index(f'ix_{table_name}_source', 'source_file', 'document_id'),
                Index(f'ix_{table_name}_strategy', 'strategy', 'confidence'),
                Index(f'ix_{table_name}_content_type', 'content_type', 'domain_type'),
                Index(f'ix_{table_name}_path', 'path'),
                Index(f'ix_{table_name}_collection', 'collection_name'),
                Index(f'ix_{table_name}_level', 'level'),
                Index(f'ix_{table_name}_dimension_value', 'dimension_value'),
                Index(f'ix_{table_name}_created', 'created_at'),

                extend_existing=True
            )

            # Create dynamic model class
            model_class = type(
                f'EmbeddingRecord_{namespace}',
                (Base,),
                {
                    '__table__': table,
                    '__repr__': lambda self: f"<EmbeddingRecord_{namespace}(id={self.id}, chunk_id='{self.chunk_id}')>"
                }
            )

            # Cache the model
            self._namespace_models[namespace] = model_class

            self.logger.debug(f"Created model for namespace '{namespace}' with table '{table_name}'")
            return model_class

    def clear_cache(self) -> None:
        """Clear the model cache."""
        with self._lock:
            self._namespace_models.clear()
            self.logger.debug("Cleared namespace model cache")


class EmbeddingSchema:
    """Manages database schema creation and migration for embeddings."""

    def __init__(self, database_url: str):
        """Initialize schema manager.

        Args:
            database_url: PostgreSQL connection URL with pgvector extension
        """
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)
        self.logger = logging.getLogger(__name__)
        self.table_factory = NamespaceTableFactory(self.engine)

    def create_extension(self) -> None:
        """Create the pgvector extension if it doesn't exist."""
        try:
            with self.engine.connect() as conn:
                # Check if pgvector extension exists
                result = conn.execute(text(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                )).scalar()

                if not result:
                    self.logger.info("Creating pgvector extension...")
                    conn.execute(text("CREATE EXTENSION vector"))
                    conn.commit()
                    self.logger.info("pgvector extension created successfully")
                else:
                    self.logger.info("pgvector extension already exists")

        except Exception as e:
            self.logger.error(f"Failed to create pgvector extension: {e}")
            raise

    def create_tables(self, namespace: str = "default") -> None:
        """Create all embedding tables and indexes.

        Args:
            namespace: Namespace for table creation (default: "default")
        """
        try:
            namespace = namespace.lower()
            self.logger.info(f"Creating database tables for namespace '{namespace}'...")

            # Get or create the model for this namespace
            model = self.table_factory.get_or_create_model(namespace)
            table_name = self.table_factory.get_table_name(namespace)

            # FIRST: Check if table exists
            with self.engine.connect() as conn:
                table_exists = conn.execute(text(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = :table_name)"
                ), {"table_name": table_name}).scalar()

                if table_exists:
                    self.logger.info(f"Table '{table_name}' already exists")
                    # Still try to create indexes in case they're missing
                    self._create_vector_indexes(namespace)
                    return

            # Create the table with IF NOT EXISTS logic
            try:
                model.__table__.create(self.engine, checkfirst=False)  # Don't check first, just try to create
                self.logger.info(f"Table created for namespace '{namespace}'")
            except Exception as table_err:
                # Check if error is due to existing constraints/indexes
                error_msg = str(table_err).lower()
                if 'already exists' in error_msg or 'duplicate' in error_msg:
                    self.logger.warning(f"Got 'already exists' error: {table_err}")
                    self.logger.warning(f"Attempting to work around by clearing metadata cache...")

                    # Clear metadata cache
                    if table_name in Base.metadata.tables:
                        Base.metadata.remove(Base.metadata.tables[table_name])
                    self.table_factory.clear_cache()

                    # Use raw SQL to create table with IF NOT EXISTS
                    self._create_table_with_raw_sql(namespace)
                else:
                    # Re-raise if it's a different error
                    raise

            # Create vector index separately (pgvector specific)
            self._create_vector_indexes(namespace)

            self.logger.info(f"Database schema created successfully for namespace '{namespace}'")

        except Exception as e:
            self.logger.error(f"Failed to create database schema for namespace '{namespace}': {e}")
            raise

    def _create_table_with_raw_sql(self, namespace: str = "default") -> None:
        """Create table using raw SQL with IF NOT EXISTS.

        This bypasses SQLAlchemy metadata cache issues.

        Args:
            namespace: Namespace for table creation
        """
        namespace = namespace.lower()
        table_name = self.table_factory.get_table_name(namespace)

        self.logger.info(f"Creating table '{table_name}' using raw SQL...")

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            embedding vector(1536),
            embedding_model VARCHAR(100) NOT NULL,
            embedding_created_at TIMESTAMP WITH TIME ZONE NOT NULL,
            chunk_id VARCHAR(4096) UNIQUE NOT NULL,
            text TEXT NOT NULL,
            text_hash VARCHAR(64) NOT NULL,
            text_length INTEGER,
            path TEXT,
            level INTEGER,
            parent_id VARCHAR(4096),
            children_ids JSONB,
            source_file TEXT,
            document_id VARCHAR(4096),
            collection_name VARCHAR(100),
            content_type VARCHAR(50),
            value_types JSONB,
            key_count INTEGER,
            strategy VARCHAR(50),
            confidence REAL,
            semantic_density REAL,
            domain_type VARCHAR(100),
            entity_types JSONB,
            performance_metrics JSONB,
            dimension_value VARCHAR(1024),
            reasoning_content JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            version VARCHAR(20) DEFAULT '1.0',
            processing_pipeline VARCHAR(50) DEFAULT 'vector_embeddings',
            CONSTRAINT uq_{table_name}_chunk_id UNIQUE (chunk_id),
            CONSTRAINT uq_{table_name}_text_hash_document UNIQUE (text_hash, document_id)
        );
        """

        # Create indexes with IF NOT EXISTS
        create_indexes_sql = f"""
        CREATE INDEX IF NOT EXISTS ix_{table_name}_source_file_document_id ON {table_name}(source_file, document_id);
        CREATE INDEX IF NOT EXISTS ix_{table_name}_strategy_confidence ON {table_name}(strategy, confidence);
        CREATE INDEX IF NOT EXISTS ix_{table_name}_content_type_domain_type ON {table_name}(content_type, domain_type);
        CREATE INDEX IF NOT EXISTS ix_{table_name}_path ON {table_name}(path);
        CREATE INDEX IF NOT EXISTS ix_{table_name}_collection ON {table_name}(collection_name);
        CREATE INDEX IF NOT EXISTS ix_{table_name}_level ON {table_name}(level);
        CREATE INDEX IF NOT EXISTS ix_{table_name}_dimension_value ON {table_name}(dimension_value);
        CREATE INDEX IF NOT EXISTS ix_{table_name}_created ON {table_name}(created_at);
        """

        try:
            with self.engine.connect() as conn:
                # Create table
                conn.execute(text(create_table_sql))
                self.logger.info(f"Table '{table_name}' created with raw SQL")

                # Create indexes
                conn.execute(text(create_indexes_sql))
                self.logger.info(f"Indexes created for '{table_name}'")

                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to create table with raw SQL: {e}")
            raise

    def _create_vector_indexes(self, namespace: str = "default") -> None:
        """Create pgvector-specific indexes.

        Args:
            namespace: Namespace for index creation (default: "default")
        """
        try:
            namespace = namespace.lower()
            table_name = self.table_factory.get_table_name(namespace)
            index_name = f"{table_name}_vector_idx"

            with self.engine.connect() as conn:
                # Check if vector index already exists
                result = conn.execute(text(
                    "SELECT EXISTS(SELECT 1 FROM pg_indexes WHERE indexname = :index_name)"
                ), {"index_name": index_name}).scalar()

                if not result:
                    self.logger.info(f"Creating vector similarity index for namespace '{namespace}'...")

                    # Create IVFFlat index for approximate nearest neighbor search with IF NOT EXISTS
                    conn.execute(text(
                        f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} "
                        "USING ivfflat (embedding vector_cosine_ops) "
                        "WITH (lists = 100)"
                    ))

                    conn.commit()
                    self.logger.info(f"Vector similarity index created for namespace '{namespace}'")
                else:
                    self.logger.info(f"Vector similarity index already exists for namespace '{namespace}'")

        except Exception as e:
            self.logger.warning(f"Failed to create vector index for namespace '{namespace}': {e}")
            # Vector index is optional for basic functionality

    def drop_tables(self) -> None:
        """Drop all embedding tables (use with caution!)."""
        try:
            self.logger.warning("Dropping database tables...")
            Base.metadata.drop_all(self.engine)
            self.logger.info("Database tables dropped")

        except Exception as e:
            self.logger.error(f"Failed to drop database tables: {e}")
            raise

    def upgrade_schema(self, target_version: Optional[str] = None) -> None:
        """Upgrade database schema to target version.

        Args:
            target_version: Target schema version (future enhancement)
        """
        # Placeholder for future migration system
        self.logger.info(f"Schema upgrade to version {target_version or 'latest'}")

        # For now, just ensure all tables and indexes exist
        self.create_tables()

    def get_schema_info(self) -> dict:
        """Get information about current schema state."""
        try:
            with self.engine.connect() as conn:
                # Check if tables exist
                tables_exist = conn.execute(text(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.tables "
                    "WHERE table_name = 'embeddings')"
                )).scalar()

                # Check if pgvector extension exists
                vector_extension = conn.execute(text(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                )).scalar()

                # Check if vector index exists
                vector_index = conn.execute(text(
                    "SELECT EXISTS(SELECT 1 FROM pg_indexes WHERE indexname = 'embeddings_vector_idx')"
                )).scalar()

                # Get row count if table exists
                row_count = 0
                if tables_exist:
                    row_count = conn.execute(text("SELECT COUNT(*) FROM embeddings")).scalar()

                return {
                    'database_url': self.database_url.split('@')[1] if '@' in self.database_url else 'configured',
                    'pgvector_extension': vector_extension,
                    'tables_exist': tables_exist,
                    'vector_index_exists': vector_index,
                    'embedding_count': row_count,
                    'schema_version': '1.0'
                }

        except Exception as e:
            self.logger.error(f"Failed to get schema info: {e}")
            return {
                'database_url': 'error',
                'pgvector_extension': False,
                'tables_exist': False,
                'vector_index_exists': False,
                'embedding_count': 0,
                'schema_version': 'unknown'
            }

    def vacuum_analyze(self) -> None:
        """Optimize database performance with VACUUM ANALYZE."""
        try:
            with self.engine.connect() as conn:
                self.logger.info("Running VACUUM ANALYZE on embeddings table...")
                conn.execute(text("VACUUM ANALYZE embeddings"))
                conn.commit()
                self.logger.info("VACUUM ANALYZE completed")

        except Exception as e:
            self.logger.error(f"Failed to run VACUUM ANALYZE: {e}")

    def create_collection_view(self, collection_name: str) -> None:
        """Create a view for a specific collection.

        Args:
            collection_name: Name of the collection
        """
        try:
            view_name = f"collection_{collection_name}"

            with self.engine.connect() as conn:
                # Drop view if exists
                conn.execute(text(f"DROP VIEW IF EXISTS {view_name}"))

                # Create view
                conn.execute(text(f"""
                    CREATE VIEW {view_name} AS
                    SELECT * FROM embeddings
                    WHERE collection_name = '{collection_name}'
                    ORDER BY created_at DESC
                """))

                conn.commit()
                self.logger.info(f"Created view '{view_name}' for collection '{collection_name}'")

        except Exception as e:
            self.logger.error(f"Failed to create collection view: {e}")
            raise

    # Namespace management methods

    def create_namespace(self, namespace: str) -> bool:
        """Create a new namespace table with schema and indexes.

        Args:
            namespace: Namespace identifier

        Returns:
            True if namespace created successfully

        Raises:
            ValueError: If namespace name is invalid
        """
        try:
            namespace = namespace.lower()
            self.table_factory.validate_namespace(namespace)

            if self.namespace_exists(namespace):
                self.logger.warning(f"Namespace '{namespace}' already exists")
                return False

            self.logger.info(f"Creating namespace '{namespace}'...")

            # Create table and indexes
            self.create_tables(namespace)

            self.logger.info(f"Namespace '{namespace}' created successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create namespace '{namespace}': {e}")
            raise

    def namespace_exists(self, namespace: str) -> bool:
        """Check if a namespace table exists.

        Args:
            namespace: Namespace identifier

        Returns:
            True if namespace exists
        """
        try:
            namespace = namespace.lower()
            return self.table_factory.table_exists(namespace)
        except Exception as e:
            self.logger.error(f"Failed to check if namespace '{namespace}' exists: {e}")
            return False

    def list_namespaces(self) -> list[Dict[str, Any]]:
        """List all namespace tables with statistics.

        Returns:
            List of namespace info dictionaries
        """
        try:
            namespaces = []

            with self.engine.connect() as conn:
                # Query all tables matching embeddings_* pattern
                result = conn.execute(text("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_name LIKE 'embeddings_%'
                    AND table_schema = 'public'
                    ORDER BY table_name
                """))

                for row in result:
                    table_name = row[0]
                    # Extract namespace from table name
                    namespace = table_name.replace('embeddings_', '')

                    # Get row count
                    count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                    row_count = count_result.scalar()

                    # Check if vector index exists
                    index_name = f"{table_name}_vector_idx"
                    index_exists = conn.execute(text(
                        "SELECT EXISTS(SELECT 1 FROM pg_indexes WHERE indexname = :index_name)"
                    ), {"index_name": index_name}).scalar()

                    namespaces.append({
                        'namespace': namespace,
                        'table_name': table_name,
                        'embedding_count': row_count,
                        'vector_index_exists': index_exists
                    })

            return namespaces

        except Exception as e:
            self.logger.error(f"Failed to list namespaces: {e}")
            return []

    def drop_namespace(self, namespace: str, confirm: bool = False) -> bool:
        """Drop a namespace table.

        Args:
            namespace: Namespace identifier
            confirm: Must be True to actually drop the table (safety check)

        Returns:
            True if namespace dropped successfully

        Raises:
            ValueError: If confirm is False
        """
        if not confirm:
            raise ValueError("Must set confirm=True to drop namespace (safety check)")

        try:
            namespace = namespace.lower()

            if not self.namespace_exists(namespace):
                self.logger.warning(f"Namespace '{namespace}' does not exist")
                return False

            table_name = self.table_factory.get_table_name(namespace)

            self.logger.warning(f"Dropping namespace '{namespace}' (table: {table_name})...")

            with self.engine.connect() as conn:
                # Drop the table (CASCADE will drop dependent objects like indexes)
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
                conn.commit()

            # Clear from cache
            if namespace in self.table_factory._namespace_models:
                del self.table_factory._namespace_models[namespace]

            self.logger.info(f"Namespace '{namespace}' dropped successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to drop namespace '{namespace}': {e}")
            raise

    def get_namespace_stats(self, namespace: str) -> Dict[str, Any]:
        """Get statistics for a specific namespace.

        Args:
            namespace: Namespace identifier

        Returns:
            Dictionary with namespace statistics
        """
        try:
            namespace = namespace.lower()

            if not self.namespace_exists(namespace):
                return {
                    'namespace': namespace,
                    'exists': False,
                    'error': 'Namespace does not exist'
                }

            table_name = self.table_factory.get_table_name(namespace)

            with self.engine.connect() as conn:
                # Get row count
                row_count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()

                # Get distinct collections
                collections = conn.execute(text(
                    f"SELECT COUNT(DISTINCT collection_name) FROM {table_name}"
                )).scalar()

                # Get table size
                table_size = conn.execute(text("""
                    SELECT pg_size_pretty(pg_total_relation_size(:table_name))
                """), {"table_name": table_name}).scalar()

                # Check vector index
                index_name = f"{table_name}_vector_idx"
                index_exists = conn.execute(text(
                    "SELECT EXISTS(SELECT 1 FROM pg_indexes WHERE indexname = :index_name)"
                ), {"index_name": index_name}).scalar()

                return {
                    'namespace': namespace,
                    'exists': True,
                    'table_name': table_name,
                    'embedding_count': row_count,
                    'collection_count': collections,
                    'table_size': table_size,
                    'vector_index_exists': index_exists
                }

        except Exception as e:
            self.logger.error(f"Failed to get stats for namespace '{namespace}': {e}")
            return {
                'namespace': namespace,
                'exists': False,
                'error': str(e)
            }