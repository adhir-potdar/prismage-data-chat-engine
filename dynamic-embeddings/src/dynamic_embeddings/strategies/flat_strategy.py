"""Flat chunking strategy for simple JSON documents."""

from typing import Dict, Any, List
from .base_strategy import BaseChunkingStrategy, DocumentChunk, ChunkMetadata


class FlatChunkingStrategy(BaseChunkingStrategy):
    """Strategy for flattening JSON into key-value chunks."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.max_keys_per_chunk = self.config.get('max_keys_per_chunk', 10)
        self.preserve_structure = self.config.get('preserve_structure', True)

    def chunk(self, json_data: Dict[str, Any], source_id: str = "document") -> List[DocumentChunk]:
        """Chunk JSON data into flat key-value pairs.

        Args:
            json_data: The JSON data to chunk
            source_id: Identifier for the source document

        Returns:
            List of flattened document chunks
        """
        chunks = []

        # Flatten the JSON structure
        flattened = self._flatten_json(json_data)

        # Group flattened keys into chunks
        chunk_groups = self._group_keys_into_chunks(flattened)

        for i, group in enumerate(chunk_groups):
            chunk_content = {}
            for key, value in group.items():
                if self.preserve_structure:
                    # Reconstruct nested structure for this key
                    self._set_nested_value(chunk_content, key, value)
                else:
                    # Keep as flat key-value
                    chunk_content[key] = value

            metadata = ChunkMetadata(
                chunk_id=self._generate_chunk_id(source_id, f"flat_{i}"),
                source_path=f"flat_chunk_{i}",
                chunk_type="flat",
                depth_level=1 if not self.preserve_structure else self._calculate_max_depth(chunk_content),
                size_bytes=self._calculate_chunk_size(chunk_content),
                key_count=len(group),
                contains_arrays=self._contains_arrays(chunk_content),
                domain_tags=self._extract_domain_tags(chunk_content)
            )

            chunk = DocumentChunk(content=chunk_content, metadata=metadata)
            if self._validate_chunk(chunk):
                chunks.append(chunk)

        return chunks

    def _flatten_json(self, obj: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten a nested JSON object."""
        items = []

        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(self._flatten_json(v, new_key, sep=sep).items())
                elif isinstance(v, list):
                    # Handle arrays specially
                    items.append((new_key, v))
                else:
                    items.append((new_key, v))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                new_key = f"{parent_key}[{i}]" if parent_key else f"[{i}]"
                if isinstance(v, (dict, list)):
                    items.extend(self._flatten_json(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
        else:
            return {parent_key: obj}

        return dict(items)

    def _group_keys_into_chunks(self, flattened: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Group flattened keys into chunks based on max_keys_per_chunk."""
        chunks = []
        current_chunk = {}

        for key, value in flattened.items():
            current_chunk[key] = value

            if len(current_chunk) >= self.max_keys_per_chunk:
                chunks.append(current_chunk)
                current_chunk = {}

        # Add remaining items
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _set_nested_value(self, obj: Dict[str, Any], key: str, value: Any) -> None:
        """Set a value in a nested dictionary using dot notation."""
        keys = key.split('.')
        current = obj

        for k in keys[:-1]:
            # Handle array indices
            if '[' in k and ']' in k:
                array_key = k.split('[')[0]
                index = int(k.split('[')[1].split(']')[0])

                if array_key not in current:
                    current[array_key] = []

                # Extend array if needed
                while len(current[array_key]) <= index:
                    current[array_key].append({})

                current = current[array_key][index]
            else:
                if k not in current:
                    current[k] = {}
                current = current[k]

        # Set the final value
        final_key = keys[-1]
        if '[' in final_key and ']' in final_key:
            array_key = final_key.split('[')[0]
            index = int(final_key.split('[')[1].split(']')[0])

            if array_key not in current:
                current[array_key] = []

            while len(current[array_key]) <= index:
                current[array_key].append(None)

            current[array_key][index] = value
        else:
            current[final_key] = value

    def _calculate_max_depth(self, obj: Any) -> int:
        """Calculate maximum depth of nested structure."""
        if isinstance(obj, dict):
            if not obj:
                return 1
            return 1 + max(self._calculate_max_depth(v) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return 1
            return 1 + max(self._calculate_max_depth(item) for item in obj)
        return 1

    def _extract_domain_tags(self, content: Dict[str, Any]) -> List[str]:
        """Extract domain-specific tags from content."""
        tags = []

        # Extract tags based on key patterns
        for key in self._get_all_keys(content):
            key_lower = key.lower()

            # Common domain patterns
            if any(pattern in key_lower for pattern in ['user', 'customer', 'person']):
                tags.append('entity:person')
            elif any(pattern in key_lower for pattern in ['product', 'item', 'sku']):
                tags.append('entity:product')
            elif any(pattern in key_lower for pattern in ['order', 'transaction', 'payment']):
                tags.append('entity:transaction')
            elif any(pattern in key_lower for pattern in ['date', 'time', 'created', 'updated']):
                tags.append('temporal')
            elif any(pattern in key_lower for pattern in ['price', 'cost', 'amount', 'value']):
                tags.append('financial')
            elif any(pattern in key_lower for pattern in ['email', 'phone', 'address']):
                tags.append('contact')

        return list(set(tags))  # Remove duplicates

    def _get_all_keys(self, obj: Any, keys: List[str] = None) -> List[str]:
        """Get all keys from nested structure."""
        if keys is None:
            keys = []

        if isinstance(obj, dict):
            for key, value in obj.items():
                keys.append(key)
                self._get_all_keys(value, keys)
        elif isinstance(obj, list):
            for item in obj:
                self._get_all_keys(item, keys)

        return keys

    def get_strategy_name(self) -> str:
        """Get the name of this strategy."""
        return "flat"