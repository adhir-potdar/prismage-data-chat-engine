"""Converts JSON chunks to embedding-ready text using proposed defaults."""

import json
import logging
from typing import Dict, Any, List


class ChunkTextConverter:
    """Converts JSON chunks to embedding-ready text using proposed defaults."""

    def __init__(self, strategy: str = "contextual_description"):
        """Initialize converter with text strategy.

        Args:
            strategy: Text conversion approach (contextual_description, key_value_pairs, etc.)
        """
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)

    def convert_chunk_to_text(self, chunk: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Convert JSON chunk to embedding-ready text.

        Args:
            chunk: JSON chunk data
            metadata: Chunk metadata with path, level info

        Returns:
            Embedding-ready text representation
        """
        if self.strategy == "contextual_description":
            return self._contextual_description(chunk, metadata)
        elif self.strategy == "key_value_pairs":
            return self._key_value_format(chunk, metadata)
        elif self.strategy == "structured_narrative":
            return self._structured_narrative(chunk, metadata)
        else:
            return self._fallback_format(chunk, metadata)

    def _contextual_description(self, chunk: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Create contextual description with path context."""
        path = metadata.get('path', '')
        level = metadata.get('level', 0)

        # Start with path context
        if path:
            context_parts = path.split('.')
            context = f"In section '{' > '.join(context_parts)}'"
        else:
            context = "At the root level"

        # Convert chunk content
        descriptions = []

        for key, value in chunk.items():
            if isinstance(value, dict):
                # Extract actual content from nested objects
                nested_content = self._extract_object_content(value)
                if nested_content:
                    descriptions.append(f"has {key} containing {nested_content}")
                else:
                    descriptions.append(f"contains a {key} object with {len(value)} properties")
            elif isinstance(value, list):
                # Extract actual content from arrays
                array_content = self._extract_array_content(value)
                if array_content:
                    descriptions.append(f"includes {key} with {array_content}")
                else:
                    descriptions.append(f"includes {key} with {len(value)} items")
            elif isinstance(value, str):
                # No truncation - preserve complete business content
                descriptions.append(f"has {key} set to '{value}'")
            elif isinstance(value, (int, float)):
                descriptions.append(f"has {key} with value {value}")
            elif isinstance(value, bool):
                descriptions.append(f"has {key} flag set to {value}")
            else:
                descriptions.append(f"contains {key} with {type(value).__name__} data")

        # Combine into readable text
        if descriptions:
            content = ", ".join(descriptions)
            return f"{context}, this structure {content}."
        else:
            return f"{context}, this is an empty structure."

    def _key_value_format(self, chunk: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Simple key-value format with path prefix."""
        path = metadata.get('path', 'root')

        pairs = []
        for key, value in chunk.items():
            if isinstance(value, dict):
                pairs.append(f"{key}: object with {len(value)} properties")
            elif isinstance(value, list):
                pairs.append(f"{key}: array with {len(value)} elements")
            else:
                value_str = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                pairs.append(f"{key}: {value_str}")

        return f"[{path}] " + " | ".join(pairs)

    def _structured_narrative(self, chunk: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Create structured narrative format."""
        path = metadata.get('path', 'document root')

        narrative_parts = [f"This document section at {path} defines"]

        # Group by data types
        objects = []
        arrays = []
        values = []

        for key, value in chunk.items():
            if isinstance(value, dict):
                objects.append(f"{key} object")
            elif isinstance(value, list):
                arrays.append(f"{key} collection")
            else:
                values.append(f"{key} property")

        parts = []
        if objects:
            parts.append(f"{', '.join(objects)} structure{'s' if len(objects) > 1 else ''}")
        if arrays:
            parts.append(f"{', '.join(arrays)} data")
        if values:
            parts.append(f"{', '.join(values)} value{'s' if len(values) > 1 else ''}")

        if parts:
            narrative_parts.append(" and ".join(parts))
            return " ".join(narrative_parts) + "."
        else:
            return f"This document section at {path} is empty."

    def _fallback_format(self, chunk: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Fallback to JSON string with path."""
        path = metadata.get('path', 'root')
        json_str = json.dumps(chunk, indent=None, separators=(',', ':'))
        return f"[{path}]: {json_str}"

    def calculate_semantic_density(self, chunk: Dict[str, Any]) -> float:
        """Calculate semantic density score (0-1)."""
        total_values = 0
        meaningful_values = 0

        def analyze_value(value):
            nonlocal total_values, meaningful_values
            total_values += 1

            if isinstance(value, str):
                # Check for meaningful text content
                if len(value.strip()) > 3 and not value.replace('_', '').replace('-', '').isdigit():
                    meaningful_values += 1
            elif isinstance(value, (dict, list)):
                meaningful_values += 1  # Structured data is considered meaningful
            elif isinstance(value, bool):
                meaningful_values += 0.5  # Partial credit for booleans
            # Numbers get no credit unless they're in meaningful ranges

        def traverse(obj):
            if isinstance(obj, dict):
                for v in obj.values():
                    if isinstance(v, (dict, list)):
                        traverse(v)
                    else:
                        analyze_value(v)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        traverse(item)
                    else:
                        analyze_value(item)

        traverse(chunk)
        return meaningful_values / max(total_values, 1)

    def _extract_object_content(self, obj: Dict[str, Any]) -> str:
        """Extract actual content from nested objects."""
        content_parts = []

        for key, value in obj.items():
            if isinstance(value, (int, float)):
                content_parts.append(f"{key} {value}")
            elif isinstance(value, str):
                # Include full string content - no truncation
                content_parts.append(f"{key} '{value}'")
            elif isinstance(value, bool):
                content_parts.append(f"{key} {value}")
            elif isinstance(value, dict):
                # Recursively extract from nested objects (max 1 level deep to avoid complexity)
                nested_content = self._extract_object_content(value)
                if nested_content:
                    content_parts.append(f"{key} with {nested_content}")
                else:
                    content_parts.append(f"{key} object")
            elif isinstance(value, list):
                array_content = self._extract_array_content(value)
                if array_content:
                    content_parts.append(f"{key} list with {array_content}")
                else:
                    content_parts.append(f"{key} list of {len(value)} items")

        return ", ".join(content_parts) if content_parts else ""

    def _extract_array_content(self, arr: List[Any]) -> str:
        """Extract actual content from arrays."""
        if not arr:
            return ""

        content_parts = []

        for item in arr:
            if isinstance(item, (int, float, str, bool)):
                content_parts.append(str(item))
            elif isinstance(item, dict):
                # For object arrays, extract key content from first few items
                obj_content = self._extract_object_content(item)
                if obj_content:
                    content_parts.append(f"object with {obj_content}")
                else:
                    content_parts.append("object")
            else:
                content_parts.append(f"{type(item).__name__}")

        return ", ".join(content_parts) if content_parts else ""