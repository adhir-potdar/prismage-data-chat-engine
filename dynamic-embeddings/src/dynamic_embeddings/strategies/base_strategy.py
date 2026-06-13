"""Base class for all chunking strategies."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json


@dataclass
class ChunkMetadata:
    """Metadata for a document chunk."""
    chunk_id: str
    source_path: str
    chunk_type: str
    depth_level: int
    parent_chunk_id: Optional[str] = None
    size_bytes: int = 0
    key_count: int = 0
    contains_arrays: bool = False
    domain_tags: List[str] = None

    def __post_init__(self):
        if self.domain_tags is None:
            self.domain_tags = []


@dataclass
class DocumentChunk:
    """Represents a chunk of a JSON document."""
    content: Dict[str, Any]
    metadata: ChunkMetadata
    text_representation: str = ""

    def __post_init__(self):
        if not self.text_representation:
            self.text_representation = json.dumps(self.content, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for storage."""
        return {
            'content': self.content,
            'metadata': {
                'chunk_id': self.metadata.chunk_id,
                'source_path': self.metadata.source_path,
                'chunk_type': self.metadata.chunk_type,
                'depth_level': self.metadata.depth_level,
                'parent_chunk_id': self.metadata.parent_chunk_id,
                'size_bytes': self.metadata.size_bytes,
                'key_count': self.metadata.key_count,
                'contains_arrays': self.metadata.contains_arrays,
                'domain_tags': self.metadata.domain_tags
            },
            'text_representation': self.text_representation
        }


class BaseChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize strategy with configuration.

        Args:
            config: Configuration dictionary for the strategy
        """
        self.config = config or {}
        self.chunk_counter = 0

    @abstractmethod
    def chunk(self, json_data: Dict[str, Any], source_id: str = "document") -> List[DocumentChunk]:
        """Chunk a JSON document into smaller pieces.

        Args:
            json_data: The JSON data to chunk
            source_id: Identifier for the source document

        Returns:
            List of document chunks
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this strategy."""
        pass

    def _generate_chunk_id(self, source_id: str, suffix: str = "") -> str:
        """Generate a unique chunk ID."""
        self.chunk_counter += 1
        base_id = f"{source_id}_chunk_{self.chunk_counter}"
        return f"{base_id}_{suffix}" if suffix else base_id

    def _calculate_chunk_size(self, content: Dict[str, Any]) -> int:
        """Calculate the size of chunk content in bytes."""
        return len(json.dumps(content).encode('utf-8'))

    def _count_keys(self, obj: Any) -> int:
        """Count total keys in a nested structure."""
        if isinstance(obj, dict):
            return len(obj) + sum(self._count_keys(v) for v in obj.values())
        elif isinstance(obj, list):
            return sum(self._count_keys(item) for item in obj)
        return 0

    def _contains_arrays(self, obj: Any) -> bool:
        """Check if object contains any arrays."""
        if isinstance(obj, list):
            return True
        elif isinstance(obj, dict):
            return any(self._contains_arrays(v) for v in obj.values())
        return False

    def _extract_path(self, obj: Dict[str, Any], path: List[str] = None) -> str:
        """Extract a path representation for the chunk."""
        if path is None:
            path = []
        return ".".join(path) if path else "root"

    def _validate_chunk(self, chunk: DocumentChunk) -> bool:
        """Validate that a chunk is properly formed."""
        if not chunk.content:
            return False
        if not chunk.metadata.chunk_id:
            return False
        if chunk.metadata.size_bytes <= 0:
            chunk.metadata.size_bytes = self._calculate_chunk_size(chunk.content)
        return True

    def get_strategy_config(self) -> Dict[str, Any]:
        """Get the current strategy configuration."""
        return {
            'strategy_name': self.get_strategy_name(),
            'config': self.config,
            'chunk_counter': self.chunk_counter
        }

    def reset_counter(self) -> None:
        """Reset the chunk counter."""
        self.chunk_counter = 0