"""Enhanced chunk structure optimized for embeddings."""

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime


@dataclass
class EmbeddingChunk:
    """Enhanced chunk structure optimized for embeddings."""

    # Core content
    text: str                           # Embedding-ready text representation
    chunk_id: str                      # Unique identifier for the chunk

    # Hierarchical metadata
    path: str                          # JSON path (e.g., "users.0.profile.details")
    level: int                         # Nesting depth
    parent_id: Optional[str] = None    # Parent chunk identifier
    children_ids: List[str] = None     # Child chunk identifiers

    # Content metadata
    content_type: str = "mixed"        # text|numeric|structured|mixed
    key_count: int = 0                 # Number of keys in original chunk
    value_types: List[str] = None      # Types of values (string, number, object, array)

    # Strategy metadata
    strategy: str = "unknown"          # Chunking strategy used
    confidence: float = 0.0            # Strategy confidence score

    # Quality metrics
    text_length: int = 0               # Character count
    semantic_density: float = 0.0      # Ratio of meaningful content

    # Source metadata
    source_file: Optional[str] = None  # Original file path
    dimension_value: Optional[str] = None  # Extracted from dimension_analyses keys (APP, AMP, etc.)
    timestamp: str = ""                # Processing timestamp

    def __post_init__(self):
        """Initialize computed fields."""
        if self.children_ids is None:
            self.children_ids = []
        if self.value_types is None:
            self.value_types = []
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
        if self.text_length == 0:
            self.text_length = len(self.text)