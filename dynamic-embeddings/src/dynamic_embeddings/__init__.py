"""Dynamic JSON Embeddings - Intelligent chunking for hierarchical JSON documents."""

from .core.dynamic_engine import DynamicChunkingEngine
from .engine.decision_engine import DecisionEngine, ChunkingStrategy
from .strategies.base_strategy import DocumentChunk, ChunkMetadata
from .config.analyzer_config import AnalyzerConfig
from .models.embedding_chunk import EmbeddingChunk
from .processors.document_processor import DocumentProcessor
from .processors.text_converter import ChunkTextConverter
from .pipelines.embedding_pipeline import EmbeddingPipeline
from .services.embedding_service import EmbeddingService, VectorEmbedding
from .services.vector_store import VectorStore
from .database.connection import DatabaseConnection
from .database.schema import EmbeddingSchema

__version__ = "1.0.0"
__author__ = "Adhir Potdar"
__email__ = "adhir.potdar@isanasystems.com"

__all__ = [
    # Core Document Processing Components
    'DynamicChunkingEngine',
    'DecisionEngine',
    'ChunkingStrategy',
    'DocumentChunk',
    'ChunkMetadata',
    'AnalyzerConfig',
    'EmbeddingChunk',
    'DocumentProcessor',
    'ChunkTextConverter',

    # Vector Embeddings Components
    'EmbeddingPipeline',
    'EmbeddingService',
    'VectorEmbedding',
    'VectorStore',
    'DatabaseConnection',
    'EmbeddingSchema'
]