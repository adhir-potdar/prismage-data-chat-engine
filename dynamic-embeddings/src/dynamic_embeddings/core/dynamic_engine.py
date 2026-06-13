"""Main dynamic engine that orchestrates JSON chunking with automatic strategy selection."""

from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import json

from ..engine.decision_engine import DecisionEngine, ChunkingStrategy
from ..strategies import (
    FlatChunkingStrategy,
    HierarchicalChunkingStrategy,
    SemanticChunkingStrategy,
    DimensionalChunkingStrategy,
    HybridChunkingStrategy,
    DocumentChunk
)
from ..config.analyzer_config import AnalyzerConfig


class DynamicChunkingEngine:
    """Main engine for dynamic JSON chunking with automatic strategy selection."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, config_name: Optional[str] = None):
        """Initialize the dynamic chunking engine.

        Args:
            config: Configuration dictionary. If None, uses default config.
            config_name: Name of configuration to load (built-in or custom). Overrides config parameter.
        """
        if config_name:
            self.config = AnalyzerConfig.get_config_by_name(config_name)
        else:
            self.config = config or AnalyzerConfig.get_default_config()
        self.decision_engine = DecisionEngine(self.config)
        self.logger = logging.getLogger(__name__)

        # Initialize all chunking strategies
        self.strategies = {
            ChunkingStrategy.FLAT: FlatChunkingStrategy(self.config.get('flat_strategy', {})),
            ChunkingStrategy.HIERARCHICAL: HierarchicalChunkingStrategy(self.config.get('hierarchical_strategy', {})),
            ChunkingStrategy.SEMANTIC: SemanticChunkingStrategy(self.config),
            ChunkingStrategy.DIMENSIONAL: DimensionalChunkingStrategy(self.config.get('dimensional_strategy', {})),
            ChunkingStrategy.HYBRID: HybridChunkingStrategy(self.config.get('hybrid_strategy', {}))
        }

        # Tracking statistics
        self.processing_stats = {
            'documents_processed': 0,
            'total_chunks_created': 0,
            'strategy_usage': {strategy.value: 0 for strategy in ChunkingStrategy},
            'processing_errors': 0
        }

    def process_document(
        self,
        json_data: Dict[str, Any],
        document_id: str = "document",
        force_strategy: Optional[ChunkingStrategy] = None
    ) -> Tuple[List[DocumentChunk], Dict[str, Any]]:
        """Process a JSON document and return chunks with metadata.

        Args:
            json_data: The JSON data to process
            document_id: Unique identifier for the document
            force_strategy: Optional strategy to force (bypasses decision engine)

        Returns:
            Tuple of (chunks, processing_metadata)
        """
        self.logger.info(f"Processing document: {document_id}")

        try:
            # Decide strategy (unless forced)
            if force_strategy:
                strategy = force_strategy
                decision_details = {
                    'chosen_strategy': strategy.value,
                    'confidence': 1.0,
                    'reasoning': f'Forced strategy: {strategy.value}',
                    'decision_factors': {}
                }
            else:
                strategy, decision_details = self.decision_engine.decide_strategy(json_data)

            # Apply the selected strategy
            chunks = self._apply_strategy(strategy, json_data, document_id)

            # Update statistics
            self._update_stats(strategy, len(chunks))

            # Prepare processing metadata
            processing_metadata = {
                'document_id': document_id,
                'strategy_used': strategy.value,
                'decision_details': decision_details,
                'chunks_created': len(chunks),
                'total_size_bytes': sum(chunk.metadata.size_bytes for chunk in chunks),
                'processing_success': True,
                'error_message': None
            }

            self.logger.info(
                f"Document processed successfully: {len(chunks)} chunks created using {strategy.value} strategy"
            )

            return chunks, processing_metadata

        except Exception as e:
            self.processing_stats['processing_errors'] += 1
            self.logger.error(f"Error processing document {document_id}: {str(e)}")

            # Return error metadata
            error_metadata = {
                'document_id': document_id,
                'strategy_used': None,
                'decision_details': {},
                'chunks_created': 0,
                'total_size_bytes': 0,
                'processing_success': False,
                'error_message': str(e)
            }

            return [], error_metadata

    def process_multiple_documents(
        self,
        documents: Dict[str, Dict[str, Any]],
        parallel_processing: bool = False
    ) -> Dict[str, Tuple[List[DocumentChunk], Dict[str, Any]]]:
        """Process multiple documents.

        Args:
            documents: Dictionary mapping document IDs to JSON data
            parallel_processing: Whether to process documents in parallel (future enhancement)

        Returns:
            Dictionary mapping document IDs to (chunks, metadata) tuples
        """
        results = {}

        for doc_id, json_data in documents.items():
            chunks, metadata = self.process_document(json_data, doc_id)
            results[doc_id] = (chunks, metadata)

        return results

    def process_from_file(self, file_path: str) -> Tuple[List[DocumentChunk], Dict[str, Any]]:
        """Process JSON document from file.

        Args:
            file_path: Path to JSON file

        Returns:
            Tuple of (chunks, processing_metadata)
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        document_id = path.stem
        return self.process_document(json_data, document_id)

    def process_from_directory(self, directory_path: str) -> Dict[str, Tuple[List[DocumentChunk], Dict[str, Any]]]:
        """Process all JSON files in a directory.

        Args:
            directory_path: Path to directory containing JSON files

        Returns:
            Dictionary mapping file names to (chunks, metadata) tuples
        """
        directory = Path(directory_path)

        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Invalid directory: {directory_path}")

        results = {}
        json_files = list(directory.glob("*.json"))

        self.logger.info(f"Processing {len(json_files)} JSON files from {directory_path}")

        for json_file in json_files:
            try:
                chunks, metadata = self.process_from_file(str(json_file))
                results[json_file.name] = (chunks, metadata)
            except Exception as e:
                self.logger.error(f"Error processing {json_file.name}: {str(e)}")
                results[json_file.name] = ([], {'error': str(e)})

        return results

    def get_strategy_recommendations(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed strategy recommendations without processing.

        Args:
            json_data: JSON data to analyze

        Returns:
            Detailed recommendations dictionary
        """
        return self.decision_engine.get_strategy_recommendations(json_data)

    def _apply_strategy(
        self,
        strategy: ChunkingStrategy,
        json_data: Dict[str, Any],
        document_id: str
    ) -> List[DocumentChunk]:
        """Apply the selected chunking strategy to the data."""
        strategy_impl = self.strategies[strategy]

        # Reset chunk counter for consistent IDs
        strategy_impl.reset_counter()

        chunks = strategy_impl.chunk(json_data, document_id)

        # Enhance chunks with additional metadata
        for chunk in chunks:
            chunk.metadata.domain_tags.append(f"engine:dynamic")
            chunk.metadata.domain_tags.append(f"strategy:{strategy.value}")

        return chunks

    def _update_stats(self, strategy: ChunkingStrategy, chunk_count: int) -> None:
        """Update processing statistics."""
        self.processing_stats['documents_processed'] += 1
        self.processing_stats['total_chunks_created'] += chunk_count
        self.processing_stats['strategy_usage'][strategy.value] += 1

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        stats = self.processing_stats.copy()

        # Add derived statistics
        if stats['documents_processed'] > 0:
            stats['average_chunks_per_document'] = (
                stats['total_chunks_created'] / stats['documents_processed']
            )
            stats['error_rate'] = (
                stats['processing_errors'] / stats['documents_processed']
            )
        else:
            stats['average_chunks_per_document'] = 0
            stats['error_rate'] = 0

        return stats

    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.processing_stats = {
            'documents_processed': 0,
            'total_chunks_created': 0,
            'strategy_usage': {strategy.value: 0 for strategy in ChunkingStrategy},
            'processing_errors': 0
        }

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update engine configuration.

        Args:
            new_config: New configuration dictionary
        """
        self.config.update(new_config)

        # Update decision engine
        self.decision_engine.update_config(new_config)

        # Reinitialize strategies with new config
        if 'flat_strategy' in new_config:
            self.strategies[ChunkingStrategy.FLAT] = FlatChunkingStrategy(new_config['flat_strategy'])

        if 'hierarchical_strategy' in new_config:
            self.strategies[ChunkingStrategy.HIERARCHICAL] = HierarchicalChunkingStrategy(
                new_config['hierarchical_strategy']
            )

        if 'semantic_strategy' in new_config or any(key in new_config for key in ['performance_keywords', 'reasoning_keywords']):
            self.strategies[ChunkingStrategy.SEMANTIC] = SemanticChunkingStrategy(self.config)

        if 'dimensional_strategy' in new_config:
            self.strategies[ChunkingStrategy.DIMENSIONAL] = DimensionalChunkingStrategy(
                new_config['dimensional_strategy']
            )

        if 'hybrid_strategy' in new_config:
            self.strategies[ChunkingStrategy.HYBRID] = HybridChunkingStrategy(new_config['hybrid_strategy'])

        self.logger.info("Engine configuration updated")

    def validate_document(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a JSON document before processing.

        Args:
            json_data: JSON data to validate

        Returns:
            Validation results
        """
        validation = {
            'valid': True,
            'warnings': [],
            'recommendations': [],
            'document_info': {}
        }

        try:
            # Basic validation
            if not isinstance(json_data, dict):
                validation['valid'] = False
                validation['warnings'].append("Root element must be a JSON object (dict)")
                return validation

            # Document size check
            doc_size = len(json.dumps(json_data))
            validation['document_info']['size_bytes'] = doc_size

            if doc_size > 10 * 1024 * 1024:  # 10MB
                validation['warnings'].append("Document is very large (>10MB), processing may be slow")

            if doc_size < 100:  # Very small
                validation['warnings'].append("Document is very small, may not benefit from chunking")

            # Structure analysis
            max_depth = self._calculate_max_depth(json_data)
            validation['document_info']['max_depth'] = max_depth

            if max_depth > 20:
                validation['warnings'].append("Document has very deep nesting (>20 levels)")

            # Content analysis
            total_keys = self._count_keys(json_data)
            validation['document_info']['total_keys'] = total_keys

            if total_keys > 10000:
                validation['warnings'].append("Document has many keys (>10,000), consider preprocessing")

            # Strategy recommendation
            strategy, _ = self.decision_engine.decide_strategy(json_data)
            validation['document_info']['recommended_strategy'] = strategy.value

            # Performance recommendations
            if doc_size > 1024 * 1024:  # 1MB
                validation['recommendations'].append("Consider using streaming processing for large documents")

            if max_depth > 10:
                validation['recommendations'].append("Deep nesting detected, hierarchical strategy recommended")

        except Exception as e:
            validation['valid'] = False
            validation['warnings'].append(f"Validation error: {str(e)}")

        return validation

    def _calculate_max_depth(self, obj: Any) -> int:
        """Calculate maximum nesting depth."""
        if isinstance(obj, dict):
            if not obj:
                return 1
            return 1 + max(self._calculate_max_depth(v) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return 1
            return 1 + max(self._calculate_max_depth(item) for item in obj)
        return 1

    def _count_keys(self, obj: Any) -> int:
        """Count total keys in nested structure."""
        if isinstance(obj, dict):
            return len(obj) + sum(self._count_keys(v) for v in obj.values())
        elif isinstance(obj, list):
            return sum(self._count_keys(item) for item in obj)
        return 0

    def export_chunks_to_json(
        self,
        chunks: List[DocumentChunk],
        output_file: str,
        include_metadata: bool = True
    ) -> None:
        """Export chunks to JSON file.

        Args:
            chunks: List of document chunks
            output_file: Path to output file
            include_metadata: Whether to include chunk metadata
        """
        export_data = []

        for chunk in chunks:
            chunk_data = {
                'content': chunk.content,
                'text_representation': chunk.text_representation
            }

            if include_metadata:
                chunk_data['metadata'] = chunk.metadata.__dict__

            export_data.append(chunk_data)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Exported {len(chunks)} chunks to {output_file}")

    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the engine configuration and capabilities."""
        return {
            'version': '1.0.0',
            'available_strategies': [strategy.value for strategy in ChunkingStrategy],
            'config_domains': list(self.config.get('domain_patterns', {}).keys()),
            'supported_file_types': ['json'],
            'processing_stats': self.get_processing_stats(),
            'decision_engine_config': self.decision_engine.get_strategy_config(),
            'available_configs': AnalyzerConfig.list_available_configs(),
            'custom_config_info': AnalyzerConfig.get_custom_config_info()
        }

    @classmethod
    def from_custom_config_file(cls, config_path: str, register_as: Optional[str] = None) -> 'DynamicChunkingEngine':
        """Create engine from custom configuration file.

        Args:
            config_path: Path to custom configuration file
            register_as: Optional name to register the configuration

        Returns:
            DynamicChunkingEngine instance
        """
        config = AnalyzerConfig.load_custom_config_from_file(config_path, register_as)
        return cls(config=config)

    @classmethod
    def from_config_name(cls, config_name: str) -> 'DynamicChunkingEngine':
        """Create engine from configuration name (built-in or custom).

        Args:
            config_name: Name of the configuration

        Returns:
            DynamicChunkingEngine instance
        """
        return cls(config_name=config_name)

    def switch_config(self, config_name: str) -> None:
        """Switch to a different configuration.

        Args:
            config_name: Name of configuration to switch to (built-in or custom)
        """
        new_config = AnalyzerConfig.get_config_by_name(config_name)
        self.update_config(new_config)
        self.logger.info(f"Switched to configuration: {config_name}")

    def load_custom_configs_from_directory(self, directory_path: str, prefix: Optional[str] = None) -> List[str]:
        """Load custom configurations from directory.

        Args:
            directory_path: Path to directory containing config files
            prefix: Optional prefix for configuration names

        Returns:
            List of loaded configuration names
        """
        return AnalyzerConfig.import_config_directory(directory_path, prefix)