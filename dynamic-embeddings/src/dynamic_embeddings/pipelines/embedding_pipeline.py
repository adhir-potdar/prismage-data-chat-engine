"""Complete embedding pipeline: JSON → Chunks → Embeddings → PGVector Storage."""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from ..processors.document_processor import DocumentProcessor
from ..services.embedding_service import EmbeddingService, VectorEmbedding
from ..services.vector_store import VectorStore
from ..database.connection import DatabaseConnection
from ..database.schema import EmbeddingSchema
from ..models.embedding_chunk import EmbeddingChunk
from ..strategies.base_strategy import DocumentChunk


class EmbeddingPipeline:
    """Complete pipeline for processing JSON documents into vector embeddings."""

    def __init__(
        self,
        database_connection: DatabaseConnection,
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-large",
        config_name: Optional[str] = None,
        namespace: str = "default"
    ):
        """Initialize the embedding pipeline.

        Args:
            database_connection: PGVector database connection
            openai_api_key: OpenAI API key for embeddings
            embedding_model: OpenAI embedding model to use
            config_name: Optional custom configuration name
            namespace: Namespace for embeddings (default: "default")
        """
        self.logger = logging.getLogger(__name__)

        # Initialize database components
        self.db_connection = database_connection
        self.namespace = namespace.lower()
        self.vector_store = VectorStore(database_connection, namespace=self.namespace)

        # Initialize Document Processing
        config_name = config_name or "default"
        self.document_processor = DocumentProcessor(config_name=config_name)

        # Initialize Vector Embedding Generation
        self.embedding_service = EmbeddingService(
            api_key=openai_api_key,
            model=embedding_model
        )

        # Initialize database schema
        self.schema = EmbeddingSchema(database_connection.database_url)

    def setup_database(self) -> Dict[str, Any]:
        """Setup database with required extensions and tables.

        Returns:
            Setup status and information
        """
        try:
            self.logger.info(f"Setting up database schema for namespace '{self.namespace}'...")

            # Test connection
            connection_info = self.db_connection.test_connection()
            if not connection_info['connected']:
                return {
                    'success': False,
                    'error': f"Database connection failed: {connection_info.get('error', 'Unknown error')}",
                    'connection_info': connection_info
                }

            # Create extension and tables for namespace
            self.schema.create_extension()
            self.schema.create_tables(namespace=self.namespace)

            # Get namespace stats
            namespace_stats = self.schema.get_namespace_stats(self.namespace)

            return {
                'success': True,
                'namespace': self.namespace,
                'connection_info': connection_info,
                'namespace_stats': namespace_stats
            }

        except Exception as e:
            self.logger.error(f"Database setup failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def process_json_file(
        self,
        json_file_path: str,
        collection_name: str = "default",
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a JSON file through the complete pipeline.

        Args:
            json_file_path: Path to JSON file to process
            collection_name: Collection name for organization
            document_id: Document identifier (auto-generated if None)

        Returns:
            Pipeline processing results and statistics
        """
        json_path = Path(json_file_path)

        if not json_path.exists():
            return {
                'success': False,
                'error': f"JSON file not found: {json_file_path}"
            }

        if document_id is None:
            document_id = json_path.stem

        self.logger.info(f"Processing JSON file: {json_file_path}")

        try:
            # Document Chunking: JSON → EmbeddingChunks
            chunking_result = self._json_to_chunks(json_file_path, document_id)

            if not chunking_result['success']:
                return chunking_result

            chunks = chunking_result['chunks']

            # Vector Generation: EmbeddingChunks → VectorEmbeddings → PGVector
            embedding_result = self._chunks_to_vectors(
                chunks, collection_name, document_id
            )

            if not embedding_result['success']:
                return embedding_result

            # Combine results
            return {
                'success': True,
                'document_id': document_id,
                'collection_name': collection_name,
                'chunking_stats': chunking_result['stats'],
                'embedding_stats': embedding_result['stats'],
                'total_embeddings': len(embedding_result['vector_embeddings']),
                'database_ids': embedding_result['database_ids']
            }

        except Exception as e:
            self.logger.error(f"Pipeline processing failed for {json_file_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'document_id': document_id,
                'collection_name': collection_name
            }

    def process_json_data(
        self,
        json_data: Dict[str, Any],
        collection_name: str = "default",
        document_id: str = "document"
    ) -> Dict[str, Any]:
        """Process JSON data dictionary through the complete pipeline.

        Args:
            json_data: JSON data as dictionary
            collection_name: Collection name for organization
            document_id: Document identifier

        Returns:
            Pipeline processing results and statistics
        """
        self.logger.info(f"Processing JSON data for document: {document_id}")

        try:
            # Document Chunking: JSON → EmbeddingChunks
            chunking_result = self._data_to_chunks(json_data, document_id)

            if not chunking_result['success']:
                return chunking_result

            chunks = chunking_result['chunks']

            # Vector Generation: EmbeddingChunks → VectorEmbeddings → PGVector
            embedding_result = self._chunks_to_vectors(
                chunks, collection_name, document_id
            )

            if not embedding_result['success']:
                return embedding_result

            # Combine results
            return {
                'success': True,
                'document_id': document_id,
                'collection_name': collection_name,
                'chunking_stats': chunking_result['stats'],
                'embedding_stats': embedding_result['stats'],
                'total_embeddings': len(embedding_result['vector_embeddings']),
                'database_ids': embedding_result['database_ids']
            }

        except Exception as e:
            self.logger.error(f"Pipeline processing failed for document {document_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'document_id': document_id,
                'collection_name': collection_name
            }

    def _json_to_chunks(
        self,
        json_file_path: str,
        document_id: str
    ) -> Dict[str, Any]:
        """Document Chunking: Process JSON file to EmbeddingChunks."""
        try:
            self.logger.info(f"Document Chunking: Processing JSON file {json_file_path}")

            result = self.document_processor.process_file(
                json_file_path,
                document_id=document_id
            )

            if not result['success']:
                return {
                    'success': False,
                    'error': f"Document chunking failed: {result.get('error', 'Unknown error')}",
                    'stats': result.get('stats', {})
                }

            chunks = result['embedding_chunks']

            self.logger.info(f"Document chunking complete: Generated {len(chunks)} chunks")

            return {
                'success': True,
                'chunks': chunks,
                'stats': {
                    'total_chunks': len(chunks),
                    'strategies_used': result['stats'].get('chunk_strategies', {}),
                    'text_conversion_methods': result['stats'].get('text_conversion_methods', {}),
                    'quality_metrics': result['stats'].get('quality_metrics', {})
                }
            }

        except Exception as e:
            self.logger.error(f"Document chunking failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _data_to_chunks(
        self,
        json_data: Dict[str, Any],
        document_id: str
    ) -> Dict[str, Any]:
        """Document Chunking: Process JSON data to EmbeddingChunks."""
        try:
            self.logger.info(f"Document Chunking: Processing JSON data for document {document_id}")

            result = self.document_processor.process_data(
                json_data,
                document_id=document_id
            )

            if not result['success']:
                return {
                    'success': False,
                    'error': f"Document chunking failed: {result.get('error', 'Unknown error')}",
                    'stats': result.get('stats', {})
                }

            chunks = result['embedding_chunks']

            self.logger.info(f"Document chunking complete: Generated {len(chunks)} chunks")

            return {
                'success': True,
                'chunks': chunks,
                'stats': {
                    'total_chunks': len(chunks),
                    'strategies_used': result['stats'].get('chunk_strategies', {}),
                    'text_conversion_methods': result['stats'].get('text_conversion_methods', {}),
                    'quality_metrics': result['stats'].get('quality_metrics', {})
                }
            }

        except Exception as e:
            self.logger.error(f"Document chunking failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _chunks_to_vectors(
        self,
        chunks: List[EmbeddingChunk],
        collection_name: str,
        document_id: str
    ) -> Dict[str, Any]:
        """Vector Generation: Convert EmbeddingChunks to VectorEmbeddings and store in PGVector."""
        try:
            self.logger.info(f"Vector Generation: Generating embeddings for {len(chunks)} chunks")

            # Check if Batch API is enabled (default: true)
            use_batch_api = os.getenv('USE_BATCH_API', 'true').lower() == 'true'

            if use_batch_api:
                self.logger.info("Using OpenAI Batch API for embedding generation (no rate limits, 50% cost savings)")

                # Use Batch API - asynchronous processing with no rate limits
                result = self.embedding_service.embed_chunks_batch_api(
                    chunks, self.vector_store, collection_name, document_id
                )

                self.logger.info(f"Batch API complete: Processed {result['total_stored']} embeddings via batch {result['batch_id']}")

                # Collect usage statistics
                embedding_info = self.embedding_service.get_embedding_info()

                return {
                    'success': True,
                    'vector_embeddings': [],  # Not kept in memory for batch processing
                    'database_ids': result['database_ids'],
                    'stats': {
                        'total_embeddings': result['total_stored'],
                        'embedding_model': embedding_info['model'],
                        'api_usage': embedding_info['usage_stats'],
                        'vector_dimensions': 1536,  # text-embedding-3-large dimensions
                        'storage_ids': result['database_ids'],
                        'batch_stats': {
                            'batch_id': result['batch_id'],
                            'cost_savings': '50%',
                            'rate_limit_free': True
                        }
                    }
                }
            else:
                # Fallback to streaming or traditional approach
                enable_streaming = os.getenv('ENABLE_STREAMING_STORAGE', 'true').lower() == 'true'

                if enable_streaming and len(chunks) > 50:
                    self.logger.info("Using streaming storage for large chunk set")

                    # Use streaming approach - process in batches with regular flushes
                    result = self.embedding_service.embed_chunks_streaming(
                        chunks, self.vector_store, collection_name, document_id
                    )

                    self.logger.info(f"Streaming storage complete: Processed {result['total_stored']} embeddings in {result['batches_processed']} batches")

                    # Collect usage statistics
                    embedding_info = self.embedding_service.get_embedding_info()

                    return {
                        'success': True,
                        'vector_embeddings': [],  # Not kept in memory for streaming
                        'database_ids': result['database_ids'],
                        'stats': {
                            'total_embeddings': result['total_stored'],
                            'embedding_model': embedding_info['model'],
                            'api_usage': embedding_info['usage_stats'],
                            'vector_dimensions': 1536,  # text-embedding-3-large dimensions
                            'storage_ids': result['database_ids'],
                            'streaming_stats': {
                                'batches_processed': result['batches_processed'],
                                'buffer_size': int(os.getenv('EMBEDDING_BUFFER_SIZE', '100')),
                                'memory_efficient': True
                            }
                        }
                    }
                else:
                    # Use traditional approach for smaller chunk sets
                    self.logger.info("Using traditional storage for small chunk set")

                    # Generate vector embeddings
                    vector_embeddings = self.embedding_service.embed_chunks(
                        chunks, collection_name, document_id
                    )

                    self.logger.info(f"Vector Generation: Generated {len(vector_embeddings)} vector embeddings")

                    # Store in PGVector database
                    self.logger.info("Vector Storage: Storing embeddings in PGVector database")

                    database_ids = self.vector_store.insert_embeddings(vector_embeddings)

                    self.logger.info(f"Vector storage complete: Stored {len(database_ids)} embeddings in database")

                    # Collect usage statistics
                    embedding_info = self.embedding_service.get_embedding_info()

                    return {
                        'success': True,
                        'vector_embeddings': vector_embeddings,
                        'database_ids': database_ids,
                        'stats': {
                            'total_embeddings': len(vector_embeddings),
                            'embedding_model': embedding_info['model'],
                            'api_usage': embedding_info['usage_stats'],
                            'vector_dimensions': len(vector_embeddings[0].embedding) if vector_embeddings else 0,
                            'storage_ids': database_ids
                        }
                    }

        except Exception as e:
            self.logger.error(f"Vector generation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def search_similar(
        self,
        query_text: str,
        collection_name: Optional[str] = None,
        limit: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search for similar embeddings using text query.

        Args:
            query_text: Text to search for
            collection_name: Optional collection filter
            limit: Maximum number of results (uses SEARCH_RESULT_LIMIT from env if None)
            similarity_threshold: Minimum similarity score (uses DEFAULT_SIMILARITY_THRESHOLD from env if None)
            filters: Additional filters (strategy, content_type, etc.)

        Returns:
            Dictionary with results, total_count, and search metadata
        """
        try:
            # Use environment defaults if parameters are not provided
            if limit is None:
                limit = int(os.getenv('SEARCH_RESULT_LIMIT', '10'))
            if similarity_threshold is None:
                similarity_threshold = float(os.getenv('DEFAULT_SIMILARITY_THRESHOLD', '0.0'))

            results, total_count = self.vector_store.similarity_search_by_text(
                query_text=query_text,
                embedding_service=self.embedding_service,
                limit=limit,
                collection_name=collection_name,
                filters=filters,
                similarity_threshold=similarity_threshold
            )

            search_info = {
                'query_text': query_text,
                'results': results,
                'total_embeddings': total_count,
                'matched_embeddings': len(results),
                'collection_name': collection_name,
                'similarity_threshold': similarity_threshold,
                'limit': limit
            }

            self.logger.info(f"Similarity search returned {len(results)} results from {total_count} total embeddings")
            return search_info

        except Exception as e:
            self.logger.error(f"Similarity search failed: {e}")
            return {
                'query_text': query_text,
                'results': [],
                'total_embeddings': 0,
                'matched_embeddings': 0,
                'collection_name': collection_name,
                'similarity_threshold': similarity_threshold,
                'limit': limit,
                'error': str(e)
            }

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics for a collection."""
        return self.vector_store.get_collection_stats(collection_name)

    def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections with their statistics."""
        return self.vector_store.list_collections()

    def delete_collection(self, collection_name: str) -> int:
        """Delete all embeddings in a collection."""
        return self.vector_store.delete_by_collection(collection_name)

    def delete_document(
        self,
        document_id: str,
        collection_name: Optional[str] = None
    ) -> int:
        """Delete all embeddings for a document."""
        return self.vector_store.delete_by_document(document_id, collection_name)

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline configuration."""
        return {
            'namespace': self.namespace,
            'database_info': self.db_connection.get_connection_info(),
            'embedding_service_info': self.embedding_service.get_embedding_info(),
            'document_processor_info': self.document_processor.get_processor_info(),
            'namespace_stats': self.schema.get_namespace_stats(self.namespace)
        }

    # Namespace management methods

    def create_namespace(self, namespace: str) -> bool:
        """Create a new namespace.

        Args:
            namespace: Namespace identifier

        Returns:
            True if namespace created successfully
        """
        return self.schema.create_namespace(namespace)

    def list_namespaces(self) -> List[Dict[str, Any]]:
        """List all namespaces with statistics.

        Returns:
            List of namespace information dictionaries
        """
        return self.schema.list_namespaces()

    def get_namespace_stats(self, namespace: str) -> Dict[str, Any]:
        """Get statistics for a specific namespace.

        Args:
            namespace: Namespace identifier

        Returns:
            Dictionary with namespace statistics
        """
        return self.schema.get_namespace_stats(namespace)

    def drop_namespace(self, namespace: str, confirm: bool = False) -> bool:
        """Drop a namespace.

        Args:
            namespace: Namespace identifier
            confirm: Must be True to actually drop (safety check)

        Returns:
            True if namespace dropped successfully
        """
        return self.schema.drop_namespace(namespace, confirm=confirm)

    def close(self) -> None:
        """Close all pipeline connections."""
        try:
            self.db_connection.close()
            self.logger.info("Pipeline connections closed")
        except Exception as e:
            self.logger.error(f"Error closing pipeline: {e}")