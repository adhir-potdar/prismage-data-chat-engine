"""Base classes for JSON analyzers."""

import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class AnalysisResult:
    """Container for analysis results."""

    # Structure metrics
    file_size_mb: float = 0.0
    max_depth: int = 0
    total_keys: int = 0
    array_count: int = 0
    object_count: int = 0
    leaf_node_count: int = 0
    branching_factor: float = 0.0

    # Structure patterns
    has_hierarchical_structure: bool = False
    has_drill_down_pattern: bool = False
    has_array_dominance: bool = False
    repeating_patterns: List[str] = field(default_factory=list)

    # Content characteristics
    domain_type: Optional[str] = None
    entity_types: Dict[str, int] = field(default_factory=dict)
    metric_patterns: List[str] = field(default_factory=list)
    temporal_data: bool = False
    geographical_data: bool = False
    performance_metrics: List[str] = field(default_factory=list)
    reasoning_content: List[str] = field(default_factory=list)

    # Business context
    metadata_sections: List[str] = field(default_factory=list)
    business_entities: List[str] = field(default_factory=list)
    configuration_data: bool = False
    analytics_data: bool = False

    # Complexity indicators
    complexity_score: float = 0.0
    estimated_chunk_count: int = 0
    recommended_strategies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "structure": {
                "file_size_mb": self.file_size_mb,
                "max_depth": self.max_depth,
                "total_keys": self.total_keys,
                "array_count": self.array_count,
                "object_count": self.object_count,
                "leaf_node_count": self.leaf_node_count,
                "branching_factor": self.branching_factor,
                "has_hierarchical_structure": self.has_hierarchical_structure,
                "has_drill_down_pattern": self.has_drill_down_pattern,
                "has_array_dominance": self.has_array_dominance,
                "repeating_patterns": self.repeating_patterns,
            },
            "content": {
                "domain_type": self.domain_type,
                "entity_types": self.entity_types,
                "metric_patterns": self.metric_patterns,
                "temporal_data": self.temporal_data,
                "geographical_data": self.geographical_data,
                "performance_metrics": self.performance_metrics,
                "reasoning_content": self.reasoning_content,
                "metadata_sections": self.metadata_sections,
                "business_entities": self.business_entities,
                "configuration_data": self.configuration_data,
                "analytics_data": self.analytics_data,
            },
            "analysis": {
                "complexity_score": self.complexity_score,
                "estimated_chunk_count": self.estimated_chunk_count,
                "recommended_strategies": self.recommended_strategies,
            }
        }


class BaseAnalyzer(ABC):
    """Base class for JSON analyzers."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize analyzer with optional configuration."""
        self.config = config or {}

    @abstractmethod
    def analyze(self, json_data: Union[Dict[str, Any], str]) -> AnalysisResult:
        """Analyze JSON data and return results."""
        pass

    def _parse_json_if_needed(self, data: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """Parse JSON string if needed."""
        if isinstance(data, str):
            try:
                return json.loads(data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON string: {e}")
        return data

    def _calculate_size_mb(self, data: Union[Dict[str, Any], str]) -> float:
        """Calculate approximate size in MB."""
        if isinstance(data, str):
            return len(data.encode('utf-8')) / (1024 * 1024)
        else:
            json_str = json.dumps(data, ensure_ascii=False)
            return len(json_str.encode('utf-8')) / (1024 * 1024)

    def _traverse_json(self, obj: Any, path: str = "$", level: int = 0) -> List[tuple]:
        """Traverse JSON structure and yield (path, value, level) tuples."""
        results = []

        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path != "$" else f"$.{key}"
                results.append((new_path, value, level))
                if isinstance(value, (dict, list)):
                    results.extend(self._traverse_json(value, new_path, level + 1))

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = f"{path}[{i}]"
                results.append((new_path, item, level))
                if isinstance(item, (dict, list)):
                    results.extend(self._traverse_json(item, new_path, level + 1))

        return results

    def _extract_text_content(self, obj: Any) -> List[str]:
        """Extract all text content from JSON structure."""
        texts = []

        if isinstance(obj, str):
            texts.append(obj)
        elif isinstance(obj, dict):
            for value in obj.values():
                texts.extend(self._extract_text_content(value))
        elif isinstance(obj, list):
            for item in obj:
                texts.extend(self._extract_text_content(item))

        return texts