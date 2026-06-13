"""Hybrid chunking strategy that combines multiple approaches."""

from typing import Dict, Any, List, Optional
from .base_strategy import BaseChunkingStrategy, DocumentChunk, ChunkMetadata
from .flat_strategy import FlatChunkingStrategy
from .hierarchical_strategy import HierarchicalChunkingStrategy
from .semantic_strategy import SemanticChunkingStrategy
from .dimensional_strategy import DimensionalChunkingStrategy


class HybridChunkingStrategy(BaseChunkingStrategy):
    """Strategy that combines multiple chunking approaches based on content analysis."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Strategy weights for different content types
        self.strategy_preferences = self.config.get('strategy_preferences', {
            'simple_content': 'flat',
            'nested_content': 'hierarchical',
            'semantic_content': 'semantic',
            'array_content': 'dimensional'
        })

        # Thresholds for strategy selection
        self.thresholds = self.config.get('thresholds', {
            'max_flat_depth': 3,
            'min_semantic_score': 2.0,
            'min_array_ratio': 0.3,
            'large_document_size': 50000
        })

        # Initialize sub-strategies
        self.flat_strategy = FlatChunkingStrategy(config.get('flat_config', {}))
        self.hierarchical_strategy = HierarchicalChunkingStrategy(config.get('hierarchical_config', {}))
        self.semantic_strategy = SemanticChunkingStrategy(config.get('semantic_config', {}))
        self.dimensional_strategy = DimensionalChunkingStrategy(config.get('dimensional_config', {}))

    def chunk(self, json_data: Dict[str, Any], source_id: str = "document") -> List[DocumentChunk]:
        """Chunk JSON data using hybrid approach.

        Args:
            json_data: The JSON data to chunk
            source_id: Identifier for the source document

        Returns:
            List of chunks created using optimal strategies for different parts
        """
        chunks = []

        # Analyze the document structure
        analysis = self._analyze_document_structure(json_data)

        # Apply different strategies to different parts
        strategy_assignments = self._assign_strategies(json_data, analysis)

        for assignment in strategy_assignments:
            sub_chunks = self._apply_strategy_to_section(
                assignment['data'],
                assignment['strategy'],
                f"{source_id}_{assignment['section']}"
            )

            # Tag chunks with hybrid context
            for chunk in sub_chunks:
                chunk.metadata.domain_tags.append(f"hybrid_strategy:{assignment['strategy']}")
                chunk.metadata.domain_tags.append(f"hybrid_section:{assignment['section']}")
                chunk.metadata.chunk_type = f"hybrid_{assignment['strategy']}"

            chunks.extend(sub_chunks)

        # Post-process chunks to optimize boundaries
        optimized_chunks = self._optimize_chunk_boundaries(chunks)

        return optimized_chunks

    def _analyze_document_structure(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze document to understand its structure and content types."""
        analysis = {
            'total_size': len(str(json_data)),
            'max_depth': self._calculate_max_depth(json_data),
            'total_keys': self._count_keys(json_data),
            'array_info': self._analyze_arrays_summary(json_data),
            'semantic_regions': self._identify_semantic_regions_summary(json_data),
            'content_distribution': self._analyze_content_distribution(json_data)
        }

        return analysis

    def _assign_strategies(self, json_data: Dict[str, Any], analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Assign optimal strategies to different sections of the document."""
        assignments = []

        # For small, simple documents - use flat strategy
        if (analysis['max_depth'] <= self.thresholds['max_flat_depth'] and
            analysis['total_size'] < self.thresholds['large_document_size'] and
            len(analysis['array_info']) == 0):

            assignments.append({
                'section': 'entire_document',
                'strategy': 'flat',
                'data': json_data,
                'reasoning': 'Simple, small document suitable for flat chunking'
            })
            return assignments

        # Identify and separate different content types
        sections = self._identify_document_sections(json_data, analysis)

        for section in sections:
            strategy = self._select_strategy_for_section(section, analysis)
            assignments.append({
                'section': section['name'],
                'strategy': strategy,
                'data': section['data'],
                'reasoning': section.get('reasoning', f'Selected {strategy} strategy')
            })

        return assignments

    def _identify_document_sections(self, json_data: Dict[str, Any], analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify distinct sections in the document that may need different strategies."""
        sections = []

        if isinstance(json_data, dict):
            for key, value in json_data.items():
                section_analysis = self._analyze_section(value, key)
                sections.append({
                    'name': key,
                    'data': {key: value},
                    'analysis': section_analysis,
                    'key': key
                })
        else:
            # Handle non-dict root objects
            sections.append({
                'name': 'root',
                'data': json_data,
                'analysis': analysis,
                'key': 'root'
            })

        return sections

    def _analyze_section(self, data: Any, section_name: str) -> Dict[str, Any]:
        """Analyze a specific section of the document."""
        return {
            'size': len(str(data)),
            'depth': self._calculate_max_depth(data),
            'key_count': self._count_keys(data),
            'has_arrays': self._contains_arrays(data),
            'array_ratio': self._calculate_array_ratio(data),
            'semantic_score': self._calculate_semantic_score_simple(data),
            'is_config_like': self._is_configuration_like(data, section_name),
            'is_tabular': self._is_tabular_structure(data)
        }

    def _select_strategy_for_section(self, section: Dict[str, Any], global_analysis: Dict[str, Any]) -> str:
        """Select the best strategy for a specific section."""
        analysis = section['analysis']

        # Large arrays should use dimensional strategy
        if analysis['array_ratio'] >= self.thresholds['min_array_ratio'] and analysis['has_arrays']:
            return 'dimensional'

        # High semantic content should use semantic strategy
        if analysis['semantic_score'] >= self.thresholds['min_semantic_score']:
            return 'semantic'

        # Deep nested structures should use hierarchical strategy
        if analysis['depth'] > self.thresholds['max_flat_depth']:
            return 'hierarchical'

        # Configuration-like content works well with flat strategy
        if analysis['is_config_like'] and analysis['depth'] <= 3:
            return 'flat'

        # Default to hierarchical for complex nested content
        if analysis['depth'] > 2:
            return 'hierarchical'

        # Simple content uses flat strategy
        return 'flat'

    def _apply_strategy_to_section(self, data: Dict[str, Any], strategy: str, source_id: str) -> List[DocumentChunk]:
        """Apply the selected strategy to a document section."""
        try:
            if strategy == 'flat':
                return self.flat_strategy.chunk(data, source_id)
            elif strategy == 'hierarchical':
                return self.hierarchical_strategy.chunk(data, source_id)
            elif strategy == 'semantic':
                return self.semantic_strategy.chunk(data, source_id)
            elif strategy == 'dimensional':
                return self.dimensional_strategy.chunk(data, source_id)
            else:
                # Fallback to hierarchical
                return self.hierarchical_strategy.chunk(data, source_id)
        except Exception as e:
            # Fallback strategy if primary fails
            return self.flat_strategy.chunk(data, source_id)

    def _optimize_chunk_boundaries(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Optimize chunk boundaries to avoid very small or very large chunks."""
        if not chunks:
            return chunks

        optimized = []

        # Merge very small chunks
        small_chunks = []
        for chunk in chunks:
            if chunk.metadata.size_bytes < 500 and chunk.metadata.key_count < 5:  # Very small
                small_chunks.append(chunk)
            else:
                # Process accumulated small chunks
                if small_chunks:
                    merged_chunk = self._merge_small_chunks(small_chunks)
                    if merged_chunk:
                        optimized.append(merged_chunk)
                    small_chunks = []

                # Split very large chunks
                if chunk.metadata.size_bytes > 100000:  # Very large (100KB)
                    split_chunks = self._split_large_chunk(chunk)
                    optimized.extend(split_chunks)
                else:
                    optimized.append(chunk)

        # Handle remaining small chunks
        if small_chunks:
            merged_chunk = self._merge_small_chunks(small_chunks)
            if merged_chunk:
                optimized.append(merged_chunk)

        return optimized

    def _merge_small_chunks(self, small_chunks: List[DocumentChunk]) -> Optional[DocumentChunk]:
        """Merge a list of small chunks into one larger chunk."""
        if not small_chunks:
            return None

        if len(small_chunks) == 1:
            return small_chunks[0]

        # Combine content
        merged_content = {}
        for i, chunk in enumerate(small_chunks):
            merged_content[f"merged_section_{i}"] = chunk.content

        # Create merged metadata
        total_size = sum(chunk.metadata.size_bytes for chunk in small_chunks)
        total_keys = sum(chunk.metadata.key_count for chunk in small_chunks)

        # Combine domain tags
        all_tags = []
        for chunk in small_chunks:
            all_tags.extend(chunk.metadata.domain_tags)
        unique_tags = list(set(all_tags))

        metadata = ChunkMetadata(
            chunk_id=self._generate_chunk_id("hybrid", "merged"),
            source_path="merged_small_chunks",
            chunk_type="hybrid_merged",
            depth_level=max(chunk.metadata.depth_level for chunk in small_chunks),
            size_bytes=total_size,
            key_count=total_keys,
            contains_arrays=any(chunk.metadata.contains_arrays for chunk in small_chunks),
            domain_tags=unique_tags[:10]  # Limit tags
        )

        return DocumentChunk(content=merged_content, metadata=metadata)

    def _split_large_chunk(self, chunk: DocumentChunk) -> List[DocumentChunk]:
        """Split a very large chunk into smaller pieces."""
        # For now, use flat strategy to split large chunks
        # This could be enhanced with more sophisticated splitting logic
        try:
            flat_chunks = self.flat_strategy.chunk(chunk.content, chunk.metadata.chunk_id)

            # Update chunk types to indicate they came from splitting
            for flat_chunk in flat_chunks:
                flat_chunk.metadata.chunk_type = f"{chunk.metadata.chunk_type}_split"
                flat_chunk.metadata.domain_tags.append("split_from_large_chunk")

            return flat_chunks
        except Exception:
            # Return original chunk if splitting fails
            return [chunk]

    def _calculate_array_ratio(self, obj: Any) -> float:
        """Calculate the ratio of array content to total content."""
        total_items = 0
        array_items = 0

        def count_items(item):
            nonlocal total_items, array_items
            total_items += 1

            if isinstance(item, list):
                array_items += 1
                for subitem in item:
                    count_items(subitem)
            elif isinstance(item, dict):
                for value in item.values():
                    count_items(value)

        count_items(obj)
        return array_items / max(total_items, 1)

    def _calculate_semantic_score_simple(self, obj: Any) -> float:
        """Simple semantic score calculation for section analysis."""
        if not isinstance(obj, (dict, list)):
            return 0.0

        text_content = str(obj).lower()

        # Simple keyword-based scoring
        semantic_keywords = [
            'description', 'reasoning', 'analysis', 'explanation', 'insight',
            'performance', 'metric', 'analysis', 'trend', 'growth'
        ]

        score = sum(1 for keyword in semantic_keywords if keyword in text_content)

        # Boost for longer text content
        if len(text_content) > 1000:
            score += 1

        return score

    def _is_configuration_like(self, data: Any, section_name: str) -> bool:
        """Check if data looks like configuration content."""
        if not isinstance(data, dict):
            return False

        section_lower = section_name.lower()
        config_indicators = ['config', 'setting', 'option', 'preference', 'parameter']

        # Check section name
        if any(indicator in section_lower for indicator in config_indicators):
            return True

        # Check key patterns
        keys_str = ' '.join(data.keys()).lower()
        if any(indicator in keys_str for indicator in config_indicators):
            return True

        return False

    def _is_tabular_structure(self, data: Any) -> bool:
        """Check if data represents tabular/matrix structure."""
        if isinstance(data, list) and len(data) > 2:
            # Check if it's an array of similar objects
            if all(isinstance(item, dict) for item in data[:3]):
                first_keys = set(data[0].keys()) if data[0] else set()
                return all(set(item.keys()) == first_keys for item in data[1:3])

        return False

    def _analyze_arrays_summary(self, obj: Any) -> List[Dict[str, Any]]:
        """Get summary information about arrays in the document."""
        arrays = []

        def find_arrays(item, path=""):
            if isinstance(item, list):
                arrays.append({
                    'path': path,
                    'size': len(item),
                    'type': 'array'
                })
                for i, subitem in enumerate(item):
                    find_arrays(subitem, f"{path}[{i}]")
            elif isinstance(item, dict):
                for key, value in item.items():
                    new_path = f"{path}.{key}" if path else key
                    find_arrays(value, new_path)

        find_arrays(obj)
        return arrays

    def _identify_semantic_regions_summary(self, obj: Any) -> Dict[str, int]:
        """Get summary of semantic regions in the document."""
        # Simplified version for hybrid strategy
        semantic_count = 0
        reasoning_count = 0

        text_content = str(obj).lower()

        if 'reasoning' in text_content or 'analysis' in text_content:
            reasoning_count = text_content.count('reasoning') + text_content.count('analysis')

        if any(term in text_content for term in ['description', 'explanation', 'insight']):
            semantic_count = 1

        return {
            'semantic_regions': semantic_count,
            'reasoning_regions': reasoning_count
        }

    def _analyze_content_distribution(self, obj: Any) -> Dict[str, float]:
        """Analyze the distribution of different content types."""
        total_chars = len(str(obj))

        return {
            'text_ratio': 0.5,  # Placeholder - could be enhanced
            'numeric_ratio': 0.3,
            'structural_ratio': 0.2
        }

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

    def get_strategy_name(self) -> str:
        """Get the name of this strategy."""
        return "hybrid"