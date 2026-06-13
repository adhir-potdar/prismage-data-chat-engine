"""OpenAI embedding service for generating vector embeddings from text chunks."""

import hashlib
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import openai
from openai import OpenAI
import os
from dataclasses import dataclass, asdict
import time
import tiktoken
import json
import tempfile
from pathlib import Path

from ..models.embedding_chunk import EmbeddingChunk


@dataclass
class VectorEmbedding:
    """Enhanced embedding structure with vector data and comprehensive metadata."""

    # Vector Data
    embedding: List[float]
    embedding_model: str
    embedding_created_at: str

    # Content Identity
    chunk_id: str
    text: str
    text_hash: str
    text_length: int

    # Hierarchical Context
    path: str
    level: int
    parent_id: Optional[str] = None
    children_ids: List[str] = None

    # Source Tracking
    source_file: Optional[str] = None
    dimension_value: Optional[str] = None  # Extracted from dimension_analyses keys
    document_id: str = "document"
    collection_name: str = "default"

    # Content Classification
    content_type: str = "mixed"
    value_types: List[str] = None
    key_count: int = 0

    # Strategy & Quality
    strategy: str = "unknown"
    confidence: float = 0.0
    semantic_density: float = 0.0

    # Search & Filtering
    domain_type: str = "general"
    entity_types: List[str] = None
    performance_metrics: List[str] = None
    reasoning_content: List[str] = None

    # Technical Metadata
    created_at: str = ""
    version: str = "1.0"
    processing_pipeline: str = "vector_embeddings"

    def __post_init__(self):
        """Initialize computed fields."""
        if self.children_ids is None:
            self.children_ids = []
        if self.value_types is None:
            self.value_types = []
        if self.entity_types is None:
            self.entity_types = []
        if self.performance_metrics is None:
            self.performance_metrics = []
        if self.reasoning_content is None:
            self.reasoning_content = []
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if not self.text_hash:
            self.text_hash = hashlib.sha256(self.text.encode('utf-8')).hexdigest()


class EmbeddingService:
    """Service for generating embeddings using OpenAI models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-large",
        batch_size: int = 100,
        max_retries: int = 3,
        timeout: int = 30
    ):
        """Initialize the embedding service.

        Args:
            api_key: OpenAI API key (if None, uses environment variable)
            model: OpenAI embedding model name
            batch_size: Number of texts to process in each batch
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout

        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Usage tracking
        self.usage_stats = {
            'total_tokens': 0,
            'total_requests': 0,
            'total_embeddings': 0,
            'failed_requests': 0,
            'last_reset': datetime.utcnow().isoformat()
        }

        # Initialize tokenizer for counting tokens
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        except KeyError:
            # Fallback to a default tokenizer if model not recognized
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Token limits for OpenAI embedding models (configurable via environment)
        self.max_tokens_per_request = int(os.getenv('MAX_TOKENS_PER_REQUEST', '8000'))
        self.max_tokens_per_text = int(os.getenv('MAX_TOKENS_PER_TEXT', '7500'))

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            # Fallback: rough estimation (4 characters per token)
            self.logger.warning(f"Token counting failed, using estimation: {e}")
            return len(text) // 4

    def validate_text_length(self, text: str) -> bool:
        """Check if text is within token limits."""
        token_count = self.count_tokens(text)
        return token_count <= self.max_tokens_per_text

    def subdivide_oversized_text(self, text: str) -> List[str]:
        """Subdivide text that exceeds token limits into smaller chunks."""
        text_tokens = self.count_tokens(text)

        if text_tokens <= self.max_tokens_per_text:
            return [text]

        # Calculate how many subdivisions we need
        num_subdivisions = (text_tokens // self.max_tokens_per_text) + 1
        self.logger.info(f"Subdividing text with {text_tokens} tokens into {num_subdivisions} smaller chunks")

        # Split text by approximate character count (rough estimation)
        chars_per_subdivision = len(text) // num_subdivisions
        subdivisions = []

        for i in range(num_subdivisions):
            start_idx = i * chars_per_subdivision

            if i == num_subdivisions - 1:
                # Last subdivision gets remaining text
                end_idx = len(text)
            else:
                end_idx = (i + 1) * chars_per_subdivision

                # Try to split at a reasonable boundary (sentence, word, etc.)
                # Look for sentence endings first
                for boundary in ['. ', '.\n', '! ', '?\n', '? ']:
                    boundary_idx = text.rfind(boundary, start_idx, end_idx + 200)
                    if boundary_idx > start_idx:
                        end_idx = boundary_idx + len(boundary)
                        break
                else:
                    # Fall back to word boundaries
                    boundary_idx = text.rfind(' ', start_idx, end_idx + 50)
                    if boundary_idx > start_idx:
                        end_idx = boundary_idx

            subdivision = text[start_idx:end_idx].strip()
            if subdivision:  # Only add non-empty subdivisions
                subdivision_tokens = self.count_tokens(subdivision)

                # If still too large, force split by characters (emergency fallback)
                if subdivision_tokens > self.max_tokens_per_text:
                    # Rough estimation: 4 chars per token
                    max_chars = self.max_tokens_per_text * 4
                    subdivision = subdivision[:max_chars]
                    self.logger.warning(f"Force-splitting text to {max_chars} chars (~{self.max_tokens_per_text} tokens)")

                subdivisions.append(subdivision)

        self.logger.info(f"Successfully subdivided into {len(subdivisions)} chunks with token counts: {[self.count_tokens(sub) for sub in subdivisions]}")
        return subdivisions

    def create_token_aware_batches(self, texts: List[str]) -> List[List[str]]:
        """Create batches that respect token limits, subdividing oversized texts."""
        # First, process all texts and subdivide any that are too large
        processed_texts = []
        for text in texts:
            subdivisions = self.subdivide_oversized_text(text)
            processed_texts.extend(subdivisions)

        # Log subdivision results
        if len(processed_texts) > len(texts):
            self.logger.info(f"Text subdivision: {len(texts)} original texts → {len(processed_texts)} processed texts")

        # Now create batches with the processed texts
        batches = []
        current_batch = []
        current_tokens = 0

        for text in processed_texts:
            text_tokens = self.count_tokens(text)

            # Check if text is still too large after subdivision
            if text_tokens > self.max_tokens_per_text:
                self.logger.error(f"Text still too large after subdivision: {text_tokens} tokens - skipping this chunk")
                continue  # Skip this chunk and continue with others

            # If adding this text would exceed the batch limit, start a new batch
            if current_tokens + text_tokens > self.max_tokens_per_request:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = [text]
                    current_tokens = text_tokens
                else:
                    # Edge case: single text that's close to the limit
                    batches.append([text])
                    current_tokens = 0
            else:
                current_batch.append(text)
                current_tokens += text_tokens

        # Add the last batch if it has content
        if current_batch:
            batches.append(current_batch)

        return batches

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        return self.generate_embeddings([text])[0]

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts with token-aware batching and rate limit pacing.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        embeddings = []

        # Create token-aware batches instead of fixed-size batches
        token_batches = self.create_token_aware_batches(texts)

        # Get pacing configuration from environment
        delay_after_batches = int(os.getenv('DELAY_AFTER_BATCHES', '5'))
        batch_delay_seconds = int(os.getenv('BATCH_DELAY_SECONDS', '2'))

        self.logger.info(f"Processing {len(texts)} texts in {len(token_batches)} token-aware batches")
        if len(token_batches) > delay_after_batches:
            self.logger.info(f"Pacing enabled: {batch_delay_seconds}s delay after every {delay_after_batches} batches")

        for batch_idx, batch in enumerate(token_batches, 1):
            total_tokens = sum(self.count_tokens(text) for text in batch)
            self.logger.debug(f"Processing batch {batch_idx}/{len(token_batches)}: {len(batch)} texts, {total_tokens} tokens")

            batch_embeddings = self._generate_batch_embeddings(batch)
            embeddings.extend(batch_embeddings)

            # Add delay after every N batches to prevent TPM bursts (but not after the last batch)
            if batch_idx % delay_after_batches == 0 and batch_idx < len(token_batches):
                self.logger.info(f"⏸️  Pacing: Adding {batch_delay_seconds}s delay after batch {batch_idx}/{len(token_batches)} to prevent rate limits")
                time.sleep(batch_delay_seconds)

        return embeddings

    def _generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts with retry logic."""

        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"Generating embeddings for batch of {len(texts)} texts (attempt {attempt + 1})")

                response = self.client.embeddings.create(
                    input=texts,
                    model=self.model,
                    dimensions=1536,  # Configure text-embedding-3-large to return 1536 dimensions
                    timeout=self.timeout
                )

                # Update usage statistics
                self.usage_stats['total_requests'] += 1
                self.usage_stats['total_embeddings'] += len(texts)
                if hasattr(response, 'usage') and response.usage:
                    self.usage_stats['total_tokens'] += response.usage.total_tokens

                # Extract embeddings
                embeddings = [data.embedding for data in response.data]

                self.logger.info(f"Successfully generated {len(embeddings)} embeddings")
                return embeddings

            except openai.RateLimitError as e:
                wait_time = min(2 ** attempt, 60)  # Exponential backoff, max 60s
                self.logger.warning(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{self.max_retries}")
                time.sleep(wait_time)

            except openai.APITimeoutError as e:
                self.logger.warning(f"Timeout on attempt {attempt + 1}/{self.max_retries}: {e}")

            except openai.APIError as e:
                self.logger.error(f"OpenAI API error on attempt {attempt + 1}/{self.max_retries}: {e}")

            except Exception as e:
                self.logger.error(f"Unexpected error on attempt {attempt + 1}/{self.max_retries}: {e}")

        # All retries failed
        self.usage_stats['failed_requests'] += 1
        raise Exception(f"Failed to generate embeddings after {self.max_retries} attempts")

    def embed_chunk(
        self,
        chunk: EmbeddingChunk,
        collection_name: str = "default",
        document_id: str = "document"
    ) -> VectorEmbedding:
        """Convert an EmbeddingChunk to a VectorEmbedding with generated embedding.

        Args:
            chunk: EmbeddingChunk from document chunking
            collection_name: Collection to store in
            document_id: Document identifier

        Returns:
            VectorEmbedding with generated embedding vector
        """
        # Generate embedding for the chunk text
        embedding_vector = self.generate_embedding(chunk.text)

        # Debug: Print dimension_value before creating VectorEmbedding
        if 'dimension_analyses' in chunk.path:
            print(f"🔍 chunk.path='{chunk.path}' | chunk.dimension_value='{getattr(chunk, 'dimension_value', None)}'")

        # Create VectorEmbedding with comprehensive metadata
        vector_embedding = VectorEmbedding(
            # Vector Data
            embedding=embedding_vector,
            embedding_model=self.model,
            embedding_created_at=datetime.utcnow().isoformat(),

            # Content Identity
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            text_hash=hashlib.sha256(chunk.text.encode('utf-8')).hexdigest(),
            text_length=chunk.text_length,

            # Hierarchical Context
            path=chunk.path,
            level=chunk.level,
            parent_id=chunk.parent_id,
            children_ids=chunk.children_ids.copy(),

            # Source Tracking
            source_file=chunk.source_file,
            dimension_value=chunk.dimension_value,
            document_id=document_id,
            collection_name=collection_name,

            # Content Classification
            content_type=chunk.content_type,
            value_types=chunk.value_types.copy(),
            key_count=chunk.key_count,

            # Strategy & Quality
            strategy=chunk.strategy,
            confidence=chunk.confidence,
            semantic_density=chunk.semantic_density,

            # Additional metadata (populated from content analysis if available)
            domain_type="general",  # Could be enhanced with domain detection
            entity_types=[],        # Could be enhanced with entity extraction
            performance_metrics=[], # Could be enhanced with metric detection
            reasoning_content=[],   # Could be enhanced with reasoning detection
        )

        self.logger.debug(f"Generated embedding for chunk {chunk.chunk_id}: {len(embedding_vector)} dimensions")

        return vector_embedding

    def embed_chunks(
        self,
        chunks: List[EmbeddingChunk],
        collection_name: str = "default",
        document_id: str = "document"
    ) -> List[VectorEmbedding]:
        """Convert multiple EmbeddingChunks to VectorEmbeddings.

        Args:
            chunks: List of EmbeddingChunks from document chunking
            collection_name: Collection to store in
            document_id: Document identifier

        Returns:
            List of VectorEmbeddings with generated embedding vectors
        """
        if not chunks:
            return []

        self.logger.info(f"Generating embeddings for {len(chunks)} chunks")

        # Filter out chunks that are too large before processing
        valid_chunks = []
        skipped_chunks = []

        for chunk in chunks:
            token_count = self.count_tokens(chunk.text)
            if token_count <= self.max_tokens_per_text:
                valid_chunks.append(chunk)
            else:
                skipped_chunks.append(chunk)
                self.logger.warning(f"Skipping chunk {chunk.chunk_id}: {token_count} tokens exceeds limit of {self.max_tokens_per_text}")

        if skipped_chunks:
            self.logger.warning(f"Skipped {len(skipped_chunks)} chunks due to token limits. Processing {len(valid_chunks)} valid chunks.")

        if not valid_chunks:
            self.logger.warning("No valid chunks to process after filtering")
            return []

        # Extract texts for batch processing (only from valid chunks)
        texts = [chunk.text for chunk in valid_chunks]

        # Generate embeddings in batches
        embedding_vectors = self.generate_embeddings(texts)

        # Create VectorEmbedding objects
        vector_embeddings = []
        for chunk, embedding_vector in zip(valid_chunks, embedding_vectors):
            # Debug logging for dimension_value
            if 'dimension_analyses' in chunk.path:
                print(f"🟢 EMBED_CHUNKS: path='{chunk.path}' | chunk.dimension_value='{chunk.dimension_value}'")

            vector_embedding = VectorEmbedding(
                # Vector Data
                embedding=embedding_vector,
                embedding_model=self.model,
                embedding_created_at=datetime.utcnow().isoformat(),

                # Content Identity
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                text_hash=hashlib.sha256(chunk.text.encode('utf-8')).hexdigest(),
                text_length=chunk.text_length,

                # Hierarchical Context
                path=chunk.path,
                level=chunk.level,
                parent_id=chunk.parent_id,
                children_ids=chunk.children_ids.copy(),

                # Source Tracking
                source_file=chunk.source_file,
                dimension_value=chunk.dimension_value,
                document_id=document_id,
                collection_name=collection_name,

                # Content Classification
                content_type=chunk.content_type,
                value_types=chunk.value_types.copy(),
                key_count=chunk.key_count,

                # Strategy & Quality
                strategy=chunk.strategy,
                confidence=chunk.confidence,
                semantic_density=chunk.semantic_density,
            )

            vector_embeddings.append(vector_embedding)

        self.logger.info(f"Successfully generated {len(vector_embeddings)} vector embeddings")

        return vector_embeddings

    def embed_chunks_streaming(
        self,
        chunks: List[EmbeddingChunk],
        vector_store,
        collection_name: str = "default",
        document_id: str = "document",
        buffer_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Convert EmbeddingChunks to VectorEmbeddings with streaming storage.

        This method processes chunks in batches and stores embeddings immediately,
        avoiding memory accumulation for large datasets.

        Args:
            chunks: List of EmbeddingChunks from document chunking
            vector_store: VectorStore instance for immediate storage
            collection_name: Collection to store in
            document_id: Document identifier
            buffer_size: Number of embeddings to buffer before flushing (defaults to env var)

        Returns:
            Dictionary with processing statistics and database IDs
        """
        if not chunks:
            return {
                'total_processed': 0,
                'total_stored': 0,
                'database_ids': [],
                'batches_processed': 0
            }

        # Get buffer size from environment or parameter
        if buffer_size is None:
            buffer_size = int(os.getenv('EMBEDDING_BUFFER_SIZE', '100'))

        self.logger.info(f"Streaming embeddings for {len(chunks)} chunks with buffer size {buffer_size}")

        database_ids = []
        total_stored = 0
        batch_count = 0
        current_buffer = []

        # Process chunks in streaming fashion
        for i in range(0, len(chunks), buffer_size):
            batch_chunks = chunks[i:i + buffer_size]
            batch_count += 1

            self.logger.info(f"Processing batch {batch_count}: {len(batch_chunks)} chunks (total progress: {i + len(batch_chunks)}/{len(chunks)})")

            # Filter out chunks that are too large before processing
            valid_batch_chunks = []
            for chunk in batch_chunks:
                token_count = self.count_tokens(chunk.text)
                if token_count <= self.max_tokens_per_text:
                    valid_batch_chunks.append(chunk)
                else:
                    self.logger.warning(f"Skipping chunk {chunk.chunk_id}: {token_count} tokens exceeds limit of {self.max_tokens_per_text}")

            if not valid_batch_chunks:
                self.logger.warning(f"No valid chunks in batch {batch_count}, skipping")
                continue

            # Extract texts for this batch (only from valid chunks)
            texts = [chunk.text for chunk in valid_batch_chunks]

            # Generate embeddings for this batch
            embedding_vectors = self.generate_embeddings(texts)

            # Create VectorEmbedding objects for this batch
            batch_vector_embeddings = []
            for chunk, embedding_vector in zip(valid_batch_chunks, embedding_vectors):
                vector_embedding = VectorEmbedding(
                    # Vector Data
                    embedding=embedding_vector,
                    embedding_model=self.model,
                    embedding_created_at=datetime.utcnow().isoformat(),

                    # Content Identity
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    text_hash=hashlib.sha256(chunk.text.encode('utf-8')).hexdigest(),
                    text_length=chunk.text_length,

                    # Hierarchical Context
                    path=chunk.path,
                    level=chunk.level,
                    parent_id=chunk.parent_id,
                    children_ids=chunk.children_ids.copy(),

                    # Source Tracking
                    source_file=chunk.source_file,
                    dimension_value=chunk.dimension_value,
                    document_id=document_id,
                    collection_name=collection_name,

                    # Content Classification
                    content_type=chunk.content_type,
                    value_types=chunk.value_types.copy(),
                    key_count=chunk.key_count,

                    # Strategy & Quality
                    strategy=chunk.strategy,
                    confidence=chunk.confidence,
                    semantic_density=chunk.semantic_density,

                    # Additional metadata
                    domain_type="general",
                    entity_types=[],
                    performance_metrics=[],
                    reasoning_content=[],
                )

                batch_vector_embeddings.append(vector_embedding)

            # Store this batch immediately
            self.logger.debug(f"Storing batch {batch_count}: {len(batch_vector_embeddings)} embeddings")
            batch_db_ids = vector_store.insert_embeddings(batch_vector_embeddings)

            # Track results
            database_ids.extend(batch_db_ids)
            total_stored += len(batch_db_ids)

            self.logger.info(f"Batch {batch_count} complete: {len(batch_db_ids)} embeddings stored (total: {total_stored}/{len(chunks)})")

            # Clear batch from memory
            del batch_vector_embeddings
            del embedding_vectors

        self.logger.info(f"Streaming complete: {total_stored} embeddings processed and stored in {batch_count} batches")

        return {
            'total_processed': len(chunks),
            'total_stored': total_stored,
            'database_ids': database_ids,
            'batches_processed': batch_count
        }

    def embed_chunks_batch_api(
        self,
        chunks: List[EmbeddingChunk],
        vector_store,
        collection_name: str = "default",
        document_id: str = "document",
        poll_interval: int = 60,
        max_wait_time: int = 3600
    ) -> Dict[str, Any]:
        """Convert EmbeddingChunks to VectorEmbeddings using OpenAI Batch API.

        This method uses the Batch API to avoid rate limits and reduce costs by 50%.
        Suitable for processing large volumes of embeddings asynchronously.

        Args:
            chunks: List of EmbeddingChunks from document chunking
            vector_store: VectorStore instance for storage
            collection_name: Collection to store in
            document_id: Document identifier
            poll_interval: Seconds to wait between status checks (default: 60)
            max_wait_time: Maximum seconds to wait for batch completion (default: 3600)

        Returns:
            Dictionary with processing statistics and database IDs
        """
        if not chunks:
            return {
                'total_processed': 0,
                'total_stored': 0,
                'database_ids': [],
                'batch_id': None
            }

        self.logger.info(f"Starting Batch API processing for {len(chunks)} chunks")

        # Filter out oversized chunks
        valid_chunks = []
        for chunk in chunks:
            token_count = self.count_tokens(chunk.text)
            if token_count <= self.max_tokens_per_text:
                valid_chunks.append(chunk)
            else:
                self.logger.warning(f"Skipping chunk {chunk.chunk_id}: {token_count} tokens exceeds limit")

        if not valid_chunks:
            self.logger.warning("No valid chunks to process after filtering")
            return {
                'total_processed': 0,
                'total_stored': 0,
                'database_ids': [],
                'batch_id': None
            }

        self.logger.info(f"Processing {len(valid_chunks)} valid chunks via Batch API")

        # Step 1: Create JSONL file with embedding requests
        batch_input_file = self._create_batch_input_file(valid_chunks)

        try:
            # Step 2: Upload file to OpenAI
            self.logger.info("Uploading batch input file to OpenAI...")
            uploaded_file = self.client.files.create(
                file=open(batch_input_file, 'rb'),
                purpose='batch'
            )
            self.logger.info(f"File uploaded: {uploaded_file.id}")

            # Step 3: Create batch job
            self.logger.info("Creating batch job...")
            batch = self.client.batches.create(
                input_file_id=uploaded_file.id,
                endpoint="/v1/embeddings",
                completion_window="24h"
            )
            batch_id = batch.id
            self.logger.info(f"Batch job created: {batch_id}")
            self.logger.info(f"Status: {batch.status}")

            # Step 4: Poll for completion
            self.logger.info(f"Polling for batch completion (checking every {poll_interval}s, max wait: {max_wait_time}s)...")
            elapsed_time = 0
            while elapsed_time < max_wait_time:
                batch = self.client.batches.retrieve(batch_id)
                status = batch.status

                self.logger.info(f"Batch status: {status} (elapsed: {elapsed_time}s)")

                if status == "completed":
                    self.logger.info("Batch processing completed!")
                    break
                elif status in ["failed", "expired", "cancelled"]:
                    error_msg = f"Batch processing {status}"
                    if hasattr(batch, 'errors') and batch.errors:
                        error_msg += f": {batch.errors}"
                    raise Exception(error_msg)

                time.sleep(poll_interval)
                elapsed_time += poll_interval

            if elapsed_time >= max_wait_time:
                raise Exception(f"Batch processing timeout after {max_wait_time}s")

            # Step 5: Download and process results
            self.logger.info("Downloading batch results...")
            output_file_id = batch.output_file_id

            if not output_file_id:
                raise Exception("Batch completed but no output file available")

            result_content = self.client.files.content(output_file_id)
            result_data = result_content.read().decode('utf-8')

            # Step 6: Parse results and create vector embeddings
            self.logger.info("Parsing batch results and creating vector embeddings...")
            vector_embeddings = self._parse_batch_results(
                result_data,
                valid_chunks,
                collection_name,
                document_id
            )

            # Step 7: Store embeddings in database
            self.logger.info(f"Storing {len(vector_embeddings)} embeddings in database...")
            database_ids = vector_store.insert_embeddings(vector_embeddings)

            # Update usage stats
            self.usage_stats['total_requests'] += 1
            self.usage_stats['total_embeddings'] += len(vector_embeddings)

            self.logger.info(f"Batch API processing complete: {len(database_ids)} embeddings stored")

            return {
                'total_processed': len(chunks),
                'total_stored': len(database_ids),
                'database_ids': database_ids,
                'batch_id': batch_id
            }

        finally:
            # Clean up temporary file
            if os.path.exists(batch_input_file):
                os.remove(batch_input_file)
                self.logger.debug(f"Cleaned up temporary file: {batch_input_file}")

    def _create_batch_input_file(self, chunks: List[EmbeddingChunk]) -> str:
        """Create JSONL file for batch API input.

        Args:
            chunks: List of EmbeddingChunks to process

        Returns:
            Path to created JSONL file
        """
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.jsonl',
            delete=False,
            encoding='utf-8'
        )

        for idx, chunk in enumerate(chunks):
            request = {
                "custom_id": f"chunk_{idx}_{chunk.chunk_id}",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {
                    "model": self.model,
                    "input": chunk.text,
                    "dimensions": 1536
                }
            }
            temp_file.write(json.dumps(request) + '\n')

        temp_file.close()
        self.logger.info(f"Created batch input file: {temp_file.name} ({len(chunks)} requests)")
        return temp_file.name

    def _parse_batch_results(
        self,
        result_data: str,
        chunks: List[EmbeddingChunk],
        collection_name: str,
        document_id: str
    ) -> List[VectorEmbedding]:
        """Parse batch API results and create VectorEmbeddings.

        Args:
            result_data: JSONL string with batch results
            chunks: Original chunks (for metadata)
            collection_name: Collection name
            document_id: Document ID

        Returns:
            List of VectorEmbeddings
        """
        # Parse JSONL results
        results = []
        for line in result_data.strip().split('\n'):
            if line:
                results.append(json.loads(line))

        # Create mapping from custom_id to result
        results_map = {}
        for result in results:
            custom_id = result.get('custom_id')
            if result.get('response', {}).get('status_code') == 200:
                embedding_data = result['response']['body']['data'][0]['embedding']
                results_map[custom_id] = embedding_data
            else:
                error = result.get('error', {}).get('message', 'Unknown error')
                self.logger.error(f"Error for {custom_id}: {error}")

        # Create VectorEmbeddings
        vector_embeddings = []
        for idx, chunk in enumerate(chunks):
            custom_id = f"chunk_{idx}_{chunk.chunk_id}"

            if custom_id not in results_map:
                self.logger.warning(f"No result found for chunk {chunk.chunk_id}, skipping")
                continue

            embedding_vector = results_map[custom_id]

            # Debug: Check chunk.dimension_value before creating VectorEmbedding
            if 'dimension_analyses' in chunk.path:
                print(f"🔍 BATCH: path='{chunk.path}' | chunk.dimension_value='{chunk.dimension_value}'")

            vector_embedding = VectorEmbedding(
                # Vector Data
                embedding=embedding_vector,
                embedding_model=self.model,
                embedding_created_at=datetime.utcnow().isoformat(),

                # Content Identity
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                text_hash=hashlib.sha256(chunk.text.encode('utf-8')).hexdigest(),
                text_length=chunk.text_length,

                # Hierarchical Context
                path=chunk.path,
                level=chunk.level,
                parent_id=chunk.parent_id,
                children_ids=chunk.children_ids.copy(),

                # Source Tracking
                source_file=chunk.source_file,
                dimension_value=chunk.dimension_value,
                document_id=document_id,
                collection_name=collection_name,

                # Content Classification
                content_type=chunk.content_type,
                value_types=chunk.value_types.copy(),
                key_count=chunk.key_count,

                # Strategy & Quality
                strategy=chunk.strategy,
                confidence=chunk.confidence,
                semantic_density=chunk.semantic_density,

                # Additional metadata
                domain_type="general",
                entity_types=[],
                performance_metrics=[],
                reasoning_content=[],
            )

            vector_embeddings.append(vector_embedding)

        self.logger.info(f"Created {len(vector_embeddings)} vector embeddings from batch results")
        return vector_embeddings

    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about the embedding service configuration."""
        return {
            'model': self.model,
            'batch_size': self.batch_size,
            'max_retries': self.max_retries,
            'timeout': self.timeout,
            'usage_stats': self.usage_stats.copy(),
            'api_key_configured': bool(self.api_key),
        }

    def reset_usage_stats(self) -> None:
        """Reset usage statistics."""
        self.usage_stats = {
            'total_tokens': 0,
            'total_requests': 0,
            'total_embeddings': 0,
            'failed_requests': 0,
            'last_reset': datetime.utcnow().isoformat()
        }

    async def generate_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        """Async version of generate_embeddings for high-throughput scenarios."""
        # For now, wrapping sync version - could be enhanced with async OpenAI client
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_embeddings, texts)

    async def embed_chunks_async(
        self,
        chunks: List[EmbeddingChunk],
        collection_name: str = "default",
        document_id: str = "document"
    ) -> List[VectorEmbedding]:
        """Async version of embed_chunks for high-throughput scenarios."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_chunks, chunks, collection_name, document_id)