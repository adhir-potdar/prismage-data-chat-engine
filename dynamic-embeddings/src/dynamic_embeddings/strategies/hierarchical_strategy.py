"""Hierarchical chunking strategy for nested JSON documents."""

from typing import Dict, Any, List, Optional
from .base_strategy import BaseChunkingStrategy, DocumentChunk, ChunkMetadata


class HierarchicalChunkingStrategy(BaseChunkingStrategy):
    """Strategy for preserving hierarchical structure in chunks."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.max_depth_per_chunk = self.config.get('max_depth_per_chunk', 3)
        self.max_size_bytes = self.config.get('max_size_bytes', 1024 * 100)  # 100KB
        self.preserve_relationships = self.config.get('preserve_relationships', True)

        # Skip large summary chunks (Level 0 & 1) that exceed embedding token limits
        self.skip_large_summaries = self.config.get('skip_large_summaries', True)
        self.max_tokens_per_chunk = self.config.get('max_tokens_per_chunk', 7600)  # OpenAI limit
        self.skip_levels = self.config.get('skip_levels', [0, 1])  # Levels to check for size

    def chunk(self, json_data: Dict[str, Any], source_id: str = "document") -> List[DocumentChunk]:
        """Chunk JSON data preserving hierarchical structure.

        Args:
            json_data: The JSON data to chunk
            source_id: Identifier for the source document

        Returns:
            List of hierarchical document chunks
        """
        chunks = []
        self._chunk_recursive(json_data, [], source_id, chunks)
        return chunks

    def _chunk_recursive(
        self,
        obj: Any,
        path: List[str],
        source_id: str,
        chunks: List[DocumentChunk],
        parent_chunk_id: Optional[str] = None,
        current_depth: int = 0
    ) -> None:
        """Recursively chunk data maintaining hierarchy."""

        if isinstance(obj, dict):
            # Check if this level should be its own chunk
            if self._should_create_chunk(obj, current_depth):
                chunk = self._create_chunk(obj, path, source_id, parent_chunk_id, current_depth)
                if chunk:
                    chunks.append(chunk)
                    parent_chunk_id = chunk.metadata.chunk_id

            # Process nested objects
            for key, value in obj.items():
                new_path = path + [key]
                if isinstance(value, (dict, list)) and value:
                    self._chunk_recursive(
                        value, new_path, source_id, chunks, parent_chunk_id, current_depth + 1
                    )

        elif isinstance(obj, list):
            # Handle arrays based on content type
            if self._is_complex_array(obj):
                # Each complex item in array becomes its own chunk
                for i, item in enumerate(obj):
                    if isinstance(item, dict):
                        array_path = path + [f"[{i}]"]
                        self._chunk_recursive(
                            item, array_path, source_id, chunks, parent_chunk_id, current_depth + 1
                        )
            else:
                # Simple array - include in parent chunk
                if path:
                    # Create a chunk for the simple array if it's large enough
                    if len(obj) > 10:  # Threshold for array chunking
                        chunk_content = {path[-1]: obj}
                        chunk = self._create_chunk(
                            chunk_content, path[:-1], source_id, parent_chunk_id, current_depth
                        )
                        if chunk:
                            chunks.append(chunk)

    def _should_create_chunk(self, obj: Dict[str, Any], current_depth: int) -> bool:
        """Determine if an object should become its own chunk."""

        # Size-based criteria
        size_bytes = self._calculate_chunk_size(obj)
        if size_bytes > self.max_size_bytes:
            return True

        # Depth-based criteria
        obj_depth = self._calculate_object_depth(obj)
        if obj_depth >= self.max_depth_per_chunk:
            return True

        # Content-based criteria
        key_count = len(obj) if isinstance(obj, dict) else 0
        if key_count >= 10:  # Large number of keys
            return True

        # Semantic criteria - objects with mixed content types
        if self._has_mixed_content(obj):
            return True

        return False

    def _create_chunk(
        self,
        content: Dict[str, Any],
        path: List[str],
        source_id: str,
        parent_chunk_id: Optional[str],
        depth_level: int
    ) -> Optional[DocumentChunk]:
        """Create a chunk from the given content."""

        if not content:
            return None

        # Skip large summary chunks (Level 0 & 1) if they exceed token limits
        if self.skip_large_summaries and depth_level in self.skip_levels:
            estimated_tokens = self._estimate_tokens(content)
            if estimated_tokens > self.max_tokens_per_chunk:
                # Skip creating this chunk - it would be too large for embedding
                return None

        path_str = self._extract_path(content, path)
        chunk_id = self._generate_chunk_id(source_id, f"hier_{len(path)}_{path_str.replace('.', '_')}")

        # Calculate metadata
        size_bytes = self._calculate_chunk_size(content)
        key_count = self._count_keys(content)
        contains_arrays = self._contains_arrays(content)
        domain_tags = self._extract_domain_tags(content, path)

        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            source_path=path_str,
            chunk_type="hierarchical",
            depth_level=depth_level,
            parent_chunk_id=parent_chunk_id,
            size_bytes=size_bytes,
            key_count=key_count,
            contains_arrays=contains_arrays,
            domain_tags=domain_tags
        )

        chunk = DocumentChunk(content=content, metadata=metadata)

        # Add hierarchical context if requested
        if self.preserve_relationships and parent_chunk_id:
            chunk.content["_hierarchy_context"] = {
                "parent_chunk": parent_chunk_id,
                "path": path_str,
                "level": depth_level
            }

        return chunk if self._validate_chunk(chunk) else None

    def _calculate_object_depth(self, obj: Any) -> int:
        """Calculate the depth of an object."""
        if isinstance(obj, dict):
            if not obj:
                return 0
            return 1 + max(self._calculate_object_depth(v) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return 0
            return 1 + max(self._calculate_object_depth(item) for item in obj)
        return 0

    def _is_complex_array(self, arr: List[Any]) -> bool:
        """Check if array contains complex objects."""
        if not arr:
            return False

        # Consider it complex if it contains dicts or nested arrays
        for item in arr[:5]:  # Check first 5 items
            if isinstance(item, (dict, list)):
                return True

        return False

    def _has_mixed_content(self, obj: Dict[str, Any]) -> bool:
        """Check if object has mixed content types (values of different types)."""
        if not isinstance(obj, dict):
            return False

        value_types = set()
        for value in obj.values():
            if isinstance(value, dict):
                value_types.add('object')
            elif isinstance(value, list):
                value_types.add('array')
            elif isinstance(value, str):
                value_types.add('string')
            elif isinstance(value, (int, float)):
                value_types.add('number')
            elif isinstance(value, bool):
                value_types.add('boolean')
            else:
                value_types.add('other')

        # Mixed if more than 2 types and contains objects/arrays
        return len(value_types) > 2 and ('object' in value_types or 'array' in value_types)

    def _estimate_tokens(self, content: Any) -> int:
        """Estimate token count for content.

        Uses rough approximation: 1 token ≈ 4 characters
        More accurate than byte count for OpenAI token limits.

        Args:
            content: Content to estimate tokens for

        Returns:
            Estimated token count
        """
        import json
        try:
            # Convert to JSON string to get character count
            json_str = json.dumps(content, ensure_ascii=False)
            char_count = len(json_str)
            # Rough approximation: 1 token ≈ 4 characters
            estimated_tokens = char_count // 4
            return estimated_tokens
        except Exception:
            # Fallback to byte-based estimation if JSON serialization fails
            return self._calculate_chunk_size(content) // 4

    def _extract_domain_tags(self, content: Dict[str, Any], path: List[str]) -> List[str]:
        """Extract domain-specific tags from content and path."""
        tags = []

        # Path-based tags
        path_str = ".".join(path).lower()
        if any(pattern in path_str for pattern in ['user', 'customer', 'client']):
            tags.append('domain:user_management')
        elif any(pattern in path_str for pattern in ['product', 'catalog', 'inventory']):
            tags.append('domain:product_catalog')
        elif any(pattern in path_str for pattern in ['order', 'transaction', 'payment']):
            tags.append('domain:transaction')
        elif any(pattern in path_str for pattern in ['config', 'setting', 'preference']):
            tags.append('domain:configuration')

        # Content-based tags
        if isinstance(content, dict):
            keys_str = " ".join(content.keys()).lower()

            # Structural tags
            if any(key in keys_str for key in ['id', 'uuid', 'identifier']):
                tags.append('structure:entity')
            if any(key in keys_str for key in ['name', 'title', 'label']):
                tags.append('structure:named_entity')
            if any(key in keys_str for key in ['date', 'time', 'created', 'updated']):
                tags.append('structure:temporal')
            if any(key in keys_str for key in ['meta', 'metadata', 'info']):
                tags.append('structure:metadata')

            # Semantic tags
            if any(key in keys_str for key in ['description', 'comment', 'note', 'reason']):
                tags.append('semantic:descriptive')
            if any(key in keys_str for key in ['metric', 'measure', 'value', 'count']):
                tags.append('semantic:quantitative')

        return list(set(tags))

    def get_strategy_name(self) -> str:
        """Get the name of this strategy."""
        return "hierarchical"