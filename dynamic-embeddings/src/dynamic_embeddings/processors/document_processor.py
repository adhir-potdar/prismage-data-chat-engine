"""Main processor orchestrating JSON document chunking pipeline."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import asdict

from ..core.dynamic_engine import DynamicChunkingEngine
from ..models.embedding_chunk import EmbeddingChunk
from .text_converter import ChunkTextConverter


class DocumentProcessor:
    """Main processor orchestrating JSON document chunking pipeline."""

    def __init__(self, config_name: str = "default"):
        """Initialize processor with configuration.

        Args:
            config_name: Configuration to use for chunking strategy selection
        """
        self.engine = DynamicChunkingEngine(config_name=config_name)
        self.text_converter = ChunkTextConverter(strategy="contextual_description")
        self.logger = logging.getLogger(__name__)

        # Quality thresholds
        self.min_text_length = 10
        self.min_semantic_density = 0.1

    def process_file(self, file_path: Union[str, Path], document_id: str = None) -> Dict[str, Any]:
        """Process JSON file through complete document chunking pipeline.

        Args:
            file_path: Path to JSON file
            document_id: Optional document identifier

        Returns:
            Dictionary with success status, embedding_chunks, and stats
        """
        file_path = Path(file_path)

        try:
            # Load JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            self.logger.info(f"Processing file: {file_path.name}")

            # Use process_data for consistent interface
            return self.process_data(json_data, document_id=document_id, source_file=str(file_path))

        except Exception as e:
            self.logger.error(f"Failed to process file {file_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'embedding_chunks': [],
                'stats': {}
            }

    def process_data(self, json_data: Dict[str, Any], document_id: str = None, source_file: Optional[str] = None) -> Dict[str, Any]:
        """Process JSON data through document chunking pipeline.

        Args:
            json_data: JSON document to process
            document_id: Optional document identifier
            source_file: Optional source file path

        Returns:
            Dictionary with success status, embedding_chunks, and stats
        """
        try:
            embedding_chunks = self._process_document_internal(json_data, source_file, document_id)
            stats = self.get_processing_stats(embedding_chunks)

            return {
                'success': True,
                'embedding_chunks': embedding_chunks,
                'stats': {
                    'chunk_strategies': stats.get('strategy_distribution', {}),
                    'text_conversion_methods': {'contextual_description': len(embedding_chunks)},
                    'quality_metrics': stats.get('quality_metrics', {})
                }
            }

        except Exception as e:
            self.logger.error(f"Document processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'embedding_chunks': [],
                'stats': {}
            }

    def _process_document_internal(self, json_data: Dict[str, Any], source_file: Optional[str] = None, document_id: str = "document") -> List[EmbeddingChunk]:
        """Internal method to process JSON document through document chunking pipeline.

        Args:
            json_data: JSON document to process
            source_file: Optional source file path
            document_id: Document identifier for unique chunk IDs

        Returns:
            List of embedding-ready chunks with metadata
        """
        # Step 1: Use existing chunking engine
        chunks, decision_metadata = self.engine.process_document(json_data, document_id=document_id)

        # Get strategy from the correct location in metadata
        strategy = decision_metadata.get('strategy_used', 'unknown')
        decision_details = decision_metadata.get('decision_details', {})
        confidence = decision_details.get('confidence', 0.0)

        self.logger.info(f"Selected strategy: {strategy} (confidence: {confidence:.2f})")

        # Step 2: Convert chunks to embedding format
        embedding_chunks = []

        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                'path': chunk.metadata.source_path,
                'level': chunk.metadata.depth_level
            }
            chunk_data = chunk.content

            # Convert to text
            text = self.text_converter.convert_chunk_to_text(chunk_data, chunk_metadata)

            # Calculate quality metrics
            semantic_density = self.text_converter.calculate_semantic_density(chunk_data)

            # Extract dimension value from path if present
            chunk_path = chunk.metadata.source_path or f'root.{i}'
            dimension_value = self._extract_dimension_value(chunk_path)

            # Debug logging for dimension_value extraction
            if 'dimension_analyses' in chunk_path and dimension_value:
                self.logger.debug(f"✓ Extracted dimension_value='{dimension_value}' from path='{chunk_path}'")
            elif 'dimension_analyses' in chunk_path and not dimension_value:
                self.logger.warning(f"✗ Failed to extract dimension_value from path='{chunk_path}'")

            # Create embedding chunk
            embedding_chunk = EmbeddingChunk(
                text=text,
                chunk_id=chunk.metadata.chunk_id,
                path=chunk_path,
                level=chunk.metadata.depth_level,
                content_type=self._detect_content_type(chunk_data),
                key_count=len(chunk_data) if isinstance(chunk_data, dict) else 0,
                value_types=self._get_value_types(chunk_data),
                strategy=strategy,
                confidence=confidence,
                semantic_density=semantic_density,
                source_file=source_file,
                dimension_value=dimension_value
            )

            # Debug: Verify dimension_value was set correctly
            if 'dimension_analyses' in chunk_path:
                self.logger.debug(f"✓ EmbeddingChunk created with dimension_value='{embedding_chunk.dimension_value}' (passed: '{dimension_value}')")

            # Quality validation
            if self._validate_chunk_quality(embedding_chunk):
                embedding_chunks.append(embedding_chunk)
            else:
                self.logger.warning(f"Chunk {i} failed quality validation, skipping")

        self.logger.info(f"Generated {len(embedding_chunks)} high-quality embedding chunks")

        return embedding_chunks

    def process_document(self, json_data: Dict[str, Any], source_file: Optional[str] = None, document_id: str = "document") -> List[EmbeddingChunk]:
        """Legacy method - process JSON document and return chunks directly.

        Args:
            json_data: JSON document to process
            source_file: Optional source file path
            document_id: Document identifier for unique chunk IDs

        Returns:
            List of embedding-ready chunks with metadata
        """
        return self._process_document_internal(json_data, source_file, document_id)

    def process_batch(self, file_paths: List[Union[str, Path]]) -> Dict[str, List[EmbeddingChunk]]:
        """Process multiple files in batch.

        Args:
            file_paths: List of JSON file paths

        Returns:
            Dictionary mapping file paths to embedding chunks
        """
        results = {}

        for file_path in file_paths:
            try:
                result = self.process_file(file_path)
                if result['success']:
                    results[str(file_path)] = result['embedding_chunks']
                else:
                    self.logger.error(f"Failed to process {file_path}: {result.get('error', 'Unknown error')}")
                    results[str(file_path)] = []
            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {e}")
                results[str(file_path)] = []

        total_chunks = sum(len(chunks) for chunks in results.values())
        self.logger.info(f"Batch processing complete: {total_chunks} total chunks from {len(file_paths)} files")

        return results

    def _detect_content_type(self, chunk: Dict[str, Any]) -> str:
        """Detect primary content type of chunk."""
        if not chunk:
            return "empty"

        types = set()
        for value in chunk.values():
            if isinstance(value, str):
                types.add("text")
            elif isinstance(value, (int, float)):
                types.add("numeric")
            elif isinstance(value, (dict, list)):
                types.add("structured")

        if len(types) == 1:
            return list(types)[0]
        else:
            return "mixed"

    def _get_value_types(self, chunk: Dict[str, Any]) -> List[str]:
        """Get list of value types in chunk."""
        types = []
        for value in chunk.values():
            if isinstance(value, str):
                types.append("string")
            elif isinstance(value, int):
                types.append("integer")
            elif isinstance(value, float):
                types.append("float")
            elif isinstance(value, bool):
                types.append("boolean")
            elif isinstance(value, dict):
                types.append("object")
            elif isinstance(value, list):
                types.append("array")
            else:
                types.append("other")

        return list(set(types))  # Remove duplicates

    def _extract_dimension_value(self, path: str) -> Optional[str]:
        """Extract dimension value from JSON path if present.

        For reasoning analysis files, paths like 'dimension_analyses.APP.metrics_analysis'
        contain dimension values (APP, AMP, DESK, etc.) that need to be extracted.

        Args:
            path: JSON path string (e.g., 'dimension_analyses.APP.metrics_analysis.Total Impressions')

        Returns:
            Dimension value if found in path after 'dimension_analyses', else None

        Examples:
            'dimension_analyses.APP.metrics_analysis' -> 'APP'
            'dimension_analyses.AMP' -> 'AMP'
            'analysis_metadata.period1' -> None
        """
        if not path or 'dimension_analyses' not in path:
            return None

        try:
            parts = path.split('.')
            # Find index of 'dimension_analyses'
            if 'dimension_analyses' in parts:
                idx = parts.index('dimension_analyses')
                # The next element should be the dimension value
                if idx + 1 < len(parts):
                    dimension_value = parts[idx + 1]
                    # Validate it's a reasonable dimension value (not empty, not a number)
                    if dimension_value and not dimension_value.isdigit():
                        return dimension_value
        except Exception as e:
            self.logger.debug(f"Failed to extract dimension from path '{path}': {e}")

        return None

    def _validate_chunk_quality(self, chunk: EmbeddingChunk) -> bool:
        """Validate chunk meets quality thresholds."""
        # Check minimum text length
        if chunk.text_length < self.min_text_length:
            return False

        # Check semantic density
        if chunk.semantic_density < self.min_semantic_density:
            return False

        # Check for meaningful content
        if not chunk.text.strip():
            return False

        return True

    def export_chunks(self, chunks: List[EmbeddingChunk], output_path: Union[str, Path], format: str = "json") -> None:
        """Export chunks to file for inspection or further processing.

        Args:
            chunks: List of embedding chunks
            output_path: Output file path
            format: Export format (json, jsonl, csv)
        """
        output_path = Path(output_path)

        if format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump([asdict(chunk) for chunk in chunks], f, indent=2, ensure_ascii=False)

        elif format == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    f.write(json.dumps(asdict(chunk), ensure_ascii=False) + '\n')

        elif format == "csv":
            import pandas as pd
            df = pd.DataFrame([asdict(chunk) for chunk in chunks])
            df.to_csv(output_path, index=False)

        else:
            raise ValueError(f"Unsupported export format: {format}")

        self.logger.info(f"Exported {len(chunks)} chunks to {output_path}")

    def get_processing_stats(self, chunks: List[EmbeddingChunk]) -> Dict[str, Any]:
        """Get processing statistics for chunks.

        Args:
            chunks: List of processed chunks

        Returns:
            Statistics dictionary
        """
        if not chunks:
            return {"total_chunks": 0}

        strategies = [chunk.strategy for chunk in chunks]
        content_types = [chunk.content_type for chunk in chunks]
        text_lengths = [chunk.text_length for chunk in chunks]
        semantic_densities = [chunk.semantic_density for chunk in chunks]

        stats = {
            "total_chunks": len(chunks),
            "avg_text_length": sum(text_lengths) / len(text_lengths),
            "avg_semantic_density": sum(semantic_densities) / len(semantic_densities),
            "strategy_distribution": {strategy: strategies.count(strategy) for strategy in set(strategies)},
            "content_type_distribution": {ctype: content_types.count(ctype) for ctype in set(content_types)},
            "quality_metrics": {
                "min_text_length": min(text_lengths),
                "max_text_length": max(text_lengths),
                "min_semantic_density": min(semantic_densities),
                "max_semantic_density": max(semantic_densities)
            }
        }

        return stats

    def get_processor_info(self) -> Dict[str, Any]:
        """Get information about the processor configuration.

        Returns:
            Processor configuration info
        """
        return {
            'config_name': getattr(self.engine, 'config_name', 'default'),
            'text_conversion_strategy': self.text_converter.strategy,
            'min_text_length': self.min_text_length,
            'min_semantic_density': self.min_semantic_density,
            'available_strategies': ['flat', 'hierarchical', 'semantic', 'dimensional', 'hybrid']
        }