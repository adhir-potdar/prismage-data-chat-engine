"""Structural analysis of JSON documents."""

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Union

from .base import AnalysisResult, BaseAnalyzer


class StructureAnalyzer(BaseAnalyzer):
    """Analyzes the structural characteristics of JSON documents."""

    def analyze(self, json_data: Union[Dict[str, Any], str]) -> AnalysisResult:
        """Analyze JSON structure and return structural metrics."""
        data = self._parse_json_if_needed(json_data)
        result = AnalysisResult()

        # Basic size metrics
        result.file_size_mb = self._calculate_size_mb(data)

        # Traverse the structure
        traversal = self._traverse_json(data)

        # Calculate depth and structure metrics
        result.max_depth = self._calculate_max_depth(traversal)
        result.total_keys = self._count_keys(data)
        result.array_count = self._count_arrays(data)
        result.object_count = self._count_objects(data)
        result.leaf_node_count = self._count_leaf_nodes(data)
        result.branching_factor = self._calculate_branching_factor(data)

        # Detect structural patterns
        result.has_hierarchical_structure = self._detect_hierarchical_structure(data)
        result.has_drill_down_pattern = self._detect_drill_down_pattern(data)
        result.has_array_dominance = self._detect_array_dominance(data)
        result.repeating_patterns = self._identify_repeating_patterns(data)

        # Calculate complexity score
        result.complexity_score = self._calculate_complexity_score(result)

        return result

    def _calculate_max_depth(self, traversal: List[tuple]) -> int:
        """Calculate maximum nesting depth."""
        if not traversal:
            return 0
        return max(level for _, _, level in traversal)

    def _count_keys(self, obj: Any, count: int = 0) -> int:
        """Count total number of keys in the JSON structure."""
        if isinstance(obj, dict):
            count += len(obj)
            for value in obj.values():
                count = self._count_keys(value, count)
        elif isinstance(obj, list):
            for item in obj:
                count = self._count_keys(item, count)
        return count

    def _count_arrays(self, obj: Any, count: int = 0) -> int:
        """Count total number of arrays."""
        if isinstance(obj, list):
            count += 1
            for item in obj:
                count = self._count_arrays(item, count)
        elif isinstance(obj, dict):
            for value in obj.values():
                count = self._count_arrays(value, count)
        return count

    def _count_objects(self, obj: Any, count: int = 0) -> int:
        """Count total number of objects."""
        if isinstance(obj, dict):
            count += 1
            for value in obj.values():
                count = self._count_objects(value, count)
        elif isinstance(obj, list):
            for item in obj:
                count = self._count_objects(item, count)
        return count

    def _count_leaf_nodes(self, obj: Any, count: int = 0) -> int:
        """Count leaf nodes (values that are not objects or arrays)."""
        if isinstance(obj, dict):
            for value in obj.values():
                count = self._count_leaf_nodes(value, count)
        elif isinstance(obj, list):
            for item in obj:
                count = self._count_leaf_nodes(item, count)
        else:
            count += 1
        return count

    def _calculate_branching_factor(self, obj: Any) -> float:
        """Calculate average branching factor."""
        object_sizes = []
        self._collect_object_sizes(obj, object_sizes)

        if not object_sizes:
            return 0.0

        return sum(object_sizes) / len(object_sizes)

    def _collect_object_sizes(self, obj: Any, sizes: List[int]) -> None:
        """Collect sizes of all objects in the structure."""
        if isinstance(obj, dict):
            sizes.append(len(obj))
            for value in obj.values():
                self._collect_object_sizes(value, sizes)
        elif isinstance(obj, list):
            for item in obj:
                self._collect_object_sizes(item, sizes)

    def _detect_hierarchical_structure(self, obj: Any) -> bool:
        """Detect if the structure is hierarchical."""
        max_depth = 0
        self._find_max_depth(obj, 0, max_depth)
        return max_depth >= 3

    def _find_max_depth(self, obj: Any, current_depth: int, max_depth: int) -> int:
        """Helper to find maximum depth."""
        if current_depth > max_depth:
            max_depth = current_depth

        if isinstance(obj, dict):
            for value in obj.values():
                max_depth = self._find_max_depth(value, current_depth + 1, max_depth)
        elif isinstance(obj, list):
            for item in obj:
                max_depth = self._find_max_depth(item, current_depth + 1, max_depth)

        return max_depth

    def _detect_drill_down_pattern(self, obj: Any) -> bool:
        """Detect hierarchical drill-down patterns common in analytics."""
        drill_down_indicators = [
            "drill_down", "breakdown", "hierarchical_analysis",
            "detailed_view", "sub_analysis", "nested_data"
        ]

        text_content = self._extract_text_content(obj)
        combined_text = " ".join(text_content).lower()

        return any(indicator in combined_text for indicator in drill_down_indicators)

    def _detect_array_dominance(self, obj: Any) -> bool:
        """Detect if arrays dominate the structure."""
        total_containers = self._count_arrays(obj) + self._count_objects(obj)
        array_count = self._count_arrays(obj)

        if total_containers == 0:
            return False

        array_ratio = array_count / total_containers
        return array_ratio > 0.6

    def _identify_repeating_patterns(self, obj: Any) -> List[str]:
        """Identify repeating structural patterns."""
        patterns = []

        # Check for repeating key patterns
        key_patterns = self._find_key_patterns(obj)
        patterns.extend(key_patterns)

        # Check for array patterns
        array_patterns = self._find_array_patterns(obj)
        patterns.extend(array_patterns)

        return list(set(patterns))

    def _find_key_patterns(self, obj: Any, path: str = "") -> List[str]:
        """Find repeating key patterns."""
        patterns = []
        key_counter = Counter()

        if isinstance(obj, dict):
            # Count key frequencies
            for key in obj.keys():
                key_counter[key] += 1

            # Find common patterns
            for key, count in key_counter.items():
                if count > 1:
                    patterns.append(f"repeated_key_{key}")

            # Recurse into nested structures
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                patterns.extend(self._find_key_patterns(value, new_path))

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = f"{path}[{i}]"
                patterns.extend(self._find_key_patterns(item, new_path))

        return patterns

    def _find_array_patterns(self, obj: Any) -> List[str]:
        """Find array-related patterns."""
        patterns = []

        if isinstance(obj, list):
            if len(obj) > 10:
                patterns.append("large_array")

            # Check if all items have similar structure
            if len(obj) > 1 and self._has_uniform_structure(obj):
                patterns.append("uniform_array")

        elif isinstance(obj, dict):
            for value in obj.values():
                patterns.extend(self._find_array_patterns(value))

        return patterns

    def _has_uniform_structure(self, arr: List[Any]) -> bool:
        """Check if array items have uniform structure."""
        if not arr:
            return True

        first_item = arr[0]
        first_type = type(first_item)
        first_keys = set(first_item.keys()) if isinstance(first_item, dict) else None

        for item in arr[1:]:
            if type(item) != first_type:
                return False
            if isinstance(item, dict) and set(item.keys()) != first_keys:
                return False

        return True

    def _calculate_complexity_score(self, result: AnalysisResult) -> float:
        """Calculate overall structural complexity score (0.0 to 1.0)."""
        factors = []

        # Size factor
        size_factor = min(result.file_size_mb / 10.0, 1.0)
        factors.append(size_factor * 0.2)

        # Depth factor
        depth_factor = min(result.max_depth / 10.0, 1.0)
        factors.append(depth_factor * 0.3)

        # Branching factor
        branching_factor = min(result.branching_factor / 20.0, 1.0)
        factors.append(branching_factor * 0.2)

        # Array complexity
        if result.has_array_dominance:
            factors.append(0.15)

        # Hierarchical complexity
        if result.has_hierarchical_structure:
            factors.append(0.1)

        # Pattern complexity
        pattern_factor = min(len(result.repeating_patterns) / 5.0, 1.0)
        factors.append(pattern_factor * 0.05)

        return sum(factors)