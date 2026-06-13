"""Dimensional chunking strategy for array-heavy and tabular JSON data."""

from typing import Dict, Any, List, Tuple, Optional
from .base_strategy import BaseChunkingStrategy, DocumentChunk, ChunkMetadata
import json


class DimensionalChunkingStrategy(BaseChunkingStrategy):
    """Strategy for chunking data with significant dimensional/tabular content."""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.max_array_size = self.config.get('max_array_size', 100)
        self.chunk_array_threshold = self.config.get('chunk_array_threshold', 50)
        self.preserve_array_structure = self.config.get('preserve_array_structure', True)
        self.group_related_arrays = self.config.get('group_related_arrays', True)

    def chunk(self, json_data: Dict[str, Any], source_id: str = "document") -> List[DocumentChunk]:
        """Chunk JSON data focusing on dimensional/array content.

        Args:
            json_data: The JSON data to chunk
            source_id: Identifier for the source document

        Returns:
            List of dimensionally-aware chunks
        """
        chunks = []

        # Identify array structures
        array_info = self._analyze_arrays(json_data)

        # Process large arrays first
        large_arrays = [info for info in array_info if info['size'] >= self.chunk_array_threshold]
        for array_info_item in large_arrays:
            array_chunks = self._chunk_large_array(array_info_item, source_id)
            chunks.extend(array_chunks)

        # Process remaining data (non-large arrays)
        remaining_data = self._remove_large_arrays(json_data, large_arrays)
        if remaining_data:
            remaining_chunks = self._chunk_remaining_data(remaining_data, source_id, array_info)
            chunks.extend(remaining_chunks)

        return chunks

    def _analyze_arrays(self, obj: Any, path: List[str] = None) -> List[Dict[str, Any]]:
        """Analyze all arrays in the JSON structure."""
        if path is None:
            path = []

        array_info = []

        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = path + [key]
                if isinstance(value, list):
                    info = self._analyze_single_array(value, new_path)
                    if info:
                        array_info.append(info)
                elif isinstance(value, dict):
                    array_info.extend(self._analyze_arrays(value, new_path))

        elif isinstance(obj, list):
            info = self._analyze_single_array(obj, path)
            if info:
                array_info.append(info)

            # Recurse into array elements
            for i, item in enumerate(obj):
                if isinstance(item, (dict, list)):
                    item_path = path + [f"[{i}]"]
                    array_info.extend(self._analyze_arrays(item, item_path))

        return array_info

    def _analyze_single_array(self, arr: List[Any], path: List[str]) -> Optional[Dict[str, Any]]:
        """Analyze a single array and return its characteristics."""
        if not arr:
            return None

        # Determine array content type
        content_type = self._classify_array_content(arr)
        homogeneous = self._is_homogeneous_array(arr)

        # Calculate dimensionality
        dimensions = self._calculate_array_dimensions(arr)

        # Detect tabular structure
        is_tabular = self._is_tabular_data(arr)

        # Extract schema for object arrays
        schema = None
        if content_type == 'object' and arr:
            schema = self._extract_array_schema(arr)

        return {
            'path': path,
            'path_string': '.'.join(path),
            'size': len(arr),
            'content_type': content_type,
            'homogeneous': homogeneous,
            'dimensions': dimensions,
            'is_tabular': is_tabular,
            'schema': schema,
            'sample_item': arr[0] if arr else None,
            'array_data': arr
        }

    def _classify_array_content(self, arr: List[Any]) -> str:
        """Classify the type of content in the array."""
        if not arr:
            return 'empty'

        # Check first few items to determine type
        sample_size = min(5, len(arr))
        types = set()

        for item in arr[:sample_size]:
            if isinstance(item, dict):
                types.add('object')
            elif isinstance(item, list):
                types.add('array')
            elif isinstance(item, str):
                types.add('string')
            elif isinstance(item, (int, float)):
                types.add('number')
            elif isinstance(item, bool):
                types.add('boolean')
            else:
                types.add('other')

        # Return dominant type
        if len(types) == 1:
            return types.pop()
        elif 'object' in types:
            return 'object'
        elif 'array' in types:
            return 'array'
        else:
            return 'mixed'

    def _is_homogeneous_array(self, arr: List[Any]) -> bool:
        """Check if array contains items of the same type and structure."""
        if not arr:
            return True

        first_type = type(arr[0])

        # For object arrays, check structural similarity
        if first_type == dict:
            first_keys = set(arr[0].keys()) if arr[0] else set()
            for item in arr[1:10]:  # Check first 10 items
                if not isinstance(item, dict) or set(item.keys()) != first_keys:
                    return False
            return True

        # For primitive arrays, check type consistency
        for item in arr[1:10]:  # Check first 10 items
            if type(item) != first_type:
                return False

        return True

    def _calculate_array_dimensions(self, arr: List[Any]) -> Tuple[int, ...]:
        """Calculate the dimensions of a potentially nested array."""
        if not arr:
            return (0,)

        dimensions = [len(arr)]

        # Check if it's a matrix (array of arrays with same length)
        if all(isinstance(item, list) for item in arr[:5]):
            sub_lengths = [len(item) for item in arr[:10] if isinstance(item, list)]
            if sub_lengths and all(length == sub_lengths[0] for length in sub_lengths):
                dimensions.append(sub_lengths[0])

        return tuple(dimensions)

    def _is_tabular_data(self, arr: List[Any]) -> bool:
        """Check if array represents tabular data."""
        if not arr or not isinstance(arr[0], dict):
            return False

        # Check if all objects have the same keys (table columns)
        if not self._is_homogeneous_array(arr):
            return False

        # Check if there are enough items to be considered tabular
        if len(arr) < 3:
            return False

        # Check if values look like table data (not too nested)
        first_item = arr[0]
        for value in first_item.values():
            if isinstance(value, (dict, list)):
                return False

        return True

    def _extract_array_schema(self, arr: List[Dict[str, Any]]) -> Dict[str, str]:
        """Extract schema information from an array of objects."""
        if not arr or not isinstance(arr[0], dict):
            return {}

        schema = {}
        sample_item = arr[0]

        for key, value in sample_item.items():
            if isinstance(value, str):
                schema[key] = 'string'
            elif isinstance(value, int):
                schema[key] = 'integer'
            elif isinstance(value, float):
                schema[key] = 'float'
            elif isinstance(value, bool):
                schema[key] = 'boolean'
            elif isinstance(value, dict):
                schema[key] = 'object'
            elif isinstance(value, list):
                schema[key] = 'array'
            else:
                schema[key] = 'unknown'

        return schema

    def _chunk_large_array(self, array_info: Dict[str, Any], source_id: str) -> List[DocumentChunk]:
        """Chunk a large array into smaller pieces."""
        chunks = []
        arr = array_info['array_data']
        path = array_info['path']

        # Calculate chunk size
        chunk_size = min(self.max_array_size, max(10, len(arr) // 10))

        # Chunk the array
        for i in range(0, len(arr), chunk_size):
            chunk_data = arr[i:i + chunk_size]

            # Create chunk content
            chunk_content = {
                'array_chunk': chunk_data,
                'array_metadata': {
                    'original_path': array_info['path_string'],
                    'content_type': array_info['content_type'],
                    'chunk_index': i // chunk_size,
                    'chunk_start': i,
                    'chunk_end': min(i + chunk_size, len(arr)),
                    'total_array_size': len(arr),
                    'is_tabular': array_info['is_tabular'],
                    'schema': array_info['schema']
                }
            }

            # Add tabular headers for tabular data
            if array_info['is_tabular'] and array_info['schema']:
                chunk_content['table_headers'] = list(array_info['schema'].keys())

            metadata = ChunkMetadata(
                chunk_id=self._generate_chunk_id(source_id, f"array_{i//chunk_size}"),
                source_path=f"{array_info['path_string']}[{i}:{min(i + chunk_size, len(arr))}]",
                chunk_type="dimensional_array",
                depth_level=len(path),
                size_bytes=self._calculate_chunk_size(chunk_content),
                key_count=len(chunk_data),
                contains_arrays=True,
                domain_tags=self._generate_array_tags(array_info)
            )

            chunk = DocumentChunk(content=chunk_content, metadata=metadata)
            if self._validate_chunk(chunk):
                chunks.append(chunk)

        return chunks

    def _remove_large_arrays(self, data: Any, large_arrays: List[Dict[str, Any]]) -> Any:
        """Remove large arrays from data structure, leaving placeholders."""
        if not large_arrays:
            return data

        # Create a deep copy and replace large arrays with references
        import copy
        result = copy.deepcopy(data)

        for array_info in large_arrays:
            path = array_info['path']
            self._replace_at_path(result, path, {
                '_large_array_reference': {
                    'original_path': array_info['path_string'],
                    'size': array_info['size'],
                    'content_type': array_info['content_type'],
                    'is_tabular': array_info['is_tabular']
                }
            })

        return result

    def _replace_at_path(self, obj: Any, path: List[str], replacement: Any) -> None:
        """Replace value at the given path with replacement."""
        if len(path) == 1:
            if isinstance(obj, dict):
                obj[path[0]] = replacement
            return

        key = path[0]
        if isinstance(obj, dict) and key in obj:
            self._replace_at_path(obj[key], path[1:], replacement)

    def _chunk_remaining_data(
        self,
        data: Dict[str, Any],
        source_id: str,
        array_info: List[Dict[str, Any]]
    ) -> List[DocumentChunk]:
        """Chunk the remaining data after removing large arrays."""
        chunks = []

        # Group small arrays and scalar data
        if self.group_related_arrays:
            grouped_data = self._group_small_arrays(data, array_info)
        else:
            grouped_data = [data]

        for i, group in enumerate(grouped_data):
            if not group:
                continue

            metadata = ChunkMetadata(
                chunk_id=self._generate_chunk_id(source_id, f"remaining_{i}"),
                source_path=f"remaining_data_{i}",
                chunk_type="dimensional_metadata",
                depth_level=1,
                size_bytes=self._calculate_chunk_size(group),
                key_count=self._count_keys(group),
                contains_arrays=self._contains_arrays(group),
                domain_tags=self._generate_metadata_tags(group)
            )

            chunk = DocumentChunk(content=group, metadata=metadata)
            if self._validate_chunk(chunk):
                chunks.append(chunk)

        return chunks

    def _group_small_arrays(self, data: Dict[str, Any], array_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group small arrays and related metadata together."""
        # For now, return the data as a single group
        # This could be enhanced to intelligently group related arrays
        return [data] if data else []

    def _generate_array_tags(self, array_info: Dict[str, Any]) -> List[str]:
        """Generate domain tags for array chunks."""
        tags = [
            f"array_type:{array_info['content_type']}",
            f"array_size:{self._size_category(array_info['size'])}"
        ]

        if array_info['is_tabular']:
            tags.append("structure:tabular")

        if array_info['homogeneous']:
            tags.append("structure:homogeneous")

        # Add schema-based tags
        if array_info['schema']:
            schema_types = set(array_info['schema'].values())
            for schema_type in schema_types:
                tags.append(f"schema:{schema_type}")

        # Content-specific tags
        path_lower = array_info['path_string'].lower()
        if any(term in path_lower for term in ['time', 'date', 'period']):
            tags.append("temporal:time_series")
        elif any(term in path_lower for term in ['metric', 'measure', 'value']):
            tags.append("quantitative:metrics")
        elif any(term in path_lower for term in ['item', 'product', 'entity']):
            tags.append("entity:collection")

        return tags

    def _generate_metadata_tags(self, data: Dict[str, Any]) -> List[str]:
        """Generate tags for metadata chunks."""
        tags = ["structure:metadata"]

        # Check for reference markers
        if self._contains_array_references(data):
            tags.append("contains:array_references")

        # Check for configuration data
        if any(key in str(data).lower() for key in ['config', 'setting', 'option']):
            tags.append("content:configuration")

        return tags

    def _contains_array_references(self, obj: Any) -> bool:
        """Check if object contains array references."""
        if isinstance(obj, dict):
            if '_large_array_reference' in obj:
                return True
            return any(self._contains_array_references(v) for v in obj.values())
        elif isinstance(obj, list):
            return any(self._contains_array_references(item) for item in obj)
        return False

    def _size_category(self, size: int) -> str:
        """Categorize array size."""
        if size < 10:
            return "small"
        elif size < 100:
            return "medium"
        elif size < 1000:
            return "large"
        else:
            return "very_large"

    def get_strategy_name(self) -> str:
        """Get the name of this strategy."""
        return "dimensional"