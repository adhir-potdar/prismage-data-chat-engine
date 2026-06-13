"""Content analysis of JSON documents for domain and semantic patterns."""

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Union
from pathlib import Path
import json

from .base import AnalysisResult, BaseAnalyzer


class ContentAnalyzer(BaseAnalyzer):
    """Analyzes the content and semantic characteristics of JSON documents."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configurable patterns and rules."""
        super().__init__(config)

        # Load patterns from config or use defaults
        self.domain_patterns = self._load_domain_patterns()
        self.metric_patterns = self._load_metric_patterns()
        self.entity_indicators = self._load_entity_indicators()
        self.content_type_patterns = self._load_content_type_patterns()

    def _load_domain_patterns(self) -> Dict[str, List[str]]:
        """Load domain classification patterns from config."""
        config_patterns = self.config.get("domain_patterns", {})

        # Default minimal patterns - can be overridden via config
        default_patterns = {
            "analytics": ["metric", "analysis", "performance", "trend"],
            "configuration": ["config", "setting", "parameter", "option"],
            "business_data": ["customer", "product", "order", "transaction"],
            "temporal": ["time", "date", "period", "timestamp"],
            "geographical": ["location", "region", "country", "geo"]
        }

        # Merge config patterns with defaults
        return {**default_patterns, **config_patterns}

    def _load_metric_patterns(self) -> Dict[str, str]:
        """Load metric detection patterns from config."""
        config_patterns = self.config.get("metric_patterns", {})

        default_patterns = {
            "percentage": r"percent|%|ratio|rate",
            "change": r"change|delta|diff|growth|decline",
            "count": r"count|total|sum|number|quantity",
            "average": r"average|mean|avg",
            "comparison": r"vs|versus|compare|period\d+"
        }

        return {**default_patterns, **config_patterns}

    def _load_entity_indicators(self) -> Dict[str, List[str]]:
        """Load entity type indicators from config."""
        config_indicators = self.config.get("entity_indicators", {})

        default_indicators = {
            "identifiers": ["id", "key", "uuid", "guid"],
            "names": ["name", "title", "label"],
            "values": ["value", "amount", "price", "cost"],
            "descriptions": ["description", "summary", "detail"],
            "categories": ["type", "category", "class", "group"]
        }

        return {**default_indicators, **config_indicators}

    def _load_content_type_patterns(self) -> Dict[str, List[str]]:
        """Load content type detection patterns from config."""
        config_patterns = self.config.get("content_type_patterns", {})

        default_patterns = {
            "temporal": [r"\d{4}-\d{2}-\d{2}", r"time", r"date", r"timestamp"],
            "numerical": [r"\d+\.?\d*", r"number", r"numeric", r"count"],
            "text": [r"description", r"comment", r"note", r"reason"],
            "metadata": [r"meta", r"info", r"header", r"config"]
        }

        return {**default_patterns, **config_patterns}

    def analyze(self, json_data: Union[Dict[str, Any], str]) -> AnalysisResult:
        """Analyze JSON content for domain patterns and semantics."""
        data = self._parse_json_if_needed(json_data)
        result = AnalysisResult()

        # Extract structure information
        keys_info = self._analyze_keys_structure(data)
        values_info = self._analyze_values_structure(data)

        # Analyze domain characteristics
        result.domain_type = self._classify_domain(keys_info, values_info)
        result.entity_types = self._identify_entity_types(keys_info, values_info)
        result.metric_patterns = self._identify_metric_patterns(keys_info, values_info)

        # Content type detection
        result.temporal_data = self._detect_content_type(keys_info, values_info, "temporal")
        result.geographical_data = self._detect_content_type(keys_info, values_info, "geographical")
        result.performance_metrics = self._detect_performance_metrics(keys_info, values_info)
        result.reasoning_content = self._detect_reasoning_content(values_info)

        # Business context
        result.metadata_sections = self._identify_metadata_sections(data)
        result.business_entities = self._extract_business_entities(data)
        result.configuration_data = self._detect_configuration_data(keys_info, values_info)
        result.analytics_data = self._detect_analytics_data(keys_info, values_info)

        return result

    def _analyze_keys_structure(self, data: Any) -> Dict[str, Any]:
        """Analyze the structure and patterns of JSON keys."""
        keys_info = {
            "all_keys": [],
            "key_patterns": Counter(),
            "key_types": defaultdict(int),
            "nested_levels": defaultdict(list),
            "key_lengths": []
        }

        def collect_keys(obj: Any, level: int = 0, path: str = ""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    keys_info["all_keys"].append(key)
                    keys_info["nested_levels"][level].append(key)
                    keys_info["key_lengths"].append(len(key))

                    # Analyze key patterns
                    key_type = self._classify_key_type(key)
                    keys_info["key_types"][key_type] += 1

                    # Analyze key patterns (camelCase, snake_case, etc.)
                    pattern = self._identify_key_pattern(key)
                    keys_info["key_patterns"][pattern] += 1

                    # Recurse into nested structures
                    new_path = f"{path}.{key}" if path else key
                    collect_keys(value, level + 1, new_path)

            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    collect_keys(item, level, f"{path}[{i}]")

        collect_keys(data)
        return keys_info

    def _analyze_values_structure(self, data: Any) -> Dict[str, Any]:
        """Analyze the structure and patterns of JSON values."""
        values_info = {
            "value_types": Counter(),
            "text_values": [],
            "numeric_values": [],
            "boolean_values": [],
            "null_values": 0,
            "text_patterns": Counter(),
            "value_lengths": []
        }

        def collect_values(obj: Any):
            if isinstance(obj, dict):
                for value in obj.values():
                    collect_values(value)
            elif isinstance(obj, list):
                for item in obj:
                    collect_values(item)
            else:
                # Leaf value
                value_type = type(obj).__name__
                values_info["value_types"][value_type] += 1

                if isinstance(obj, str):
                    values_info["text_values"].append(obj)
                    values_info["value_lengths"].append(len(obj))

                    # Analyze text patterns
                    pattern = self._identify_text_pattern(obj)
                    values_info["text_patterns"][pattern] += 1

                elif isinstance(obj, (int, float)):
                    values_info["numeric_values"].append(obj)
                elif isinstance(obj, bool):
                    values_info["boolean_values"].append(obj)
                elif obj is None:
                    values_info["null_values"] += 1

        collect_values(data)
        return values_info

    def _classify_key_type(self, key: str) -> str:
        """Classify the type of a JSON key."""
        key_lower = key.lower()

        # Check against configured entity indicators
        for entity_type, indicators in self.entity_indicators.items():
            if any(indicator in key_lower for indicator in indicators):
                return entity_type

        # Additional classification based on patterns
        if re.match(r'^\d+$', key):
            return "numeric_key"
        elif key.isupper():
            return "constant"
        elif '_' in key:
            return "snake_case"
        elif re.match(r'^[a-z][a-zA-Z0-9]*$', key):
            return "camelCase"
        else:
            return "other"

    def _identify_key_pattern(self, key: str) -> str:
        """Identify naming pattern of a key."""
        if '_' in key:
            return "snake_case"
        elif re.match(r'^[a-z][a-zA-Z0-9]*[A-Z]', key):
            return "camelCase"
        elif key.isupper():
            return "UPPER_CASE"
        elif key.islower():
            return "lowercase"
        elif '-' in key:
            return "kebab-case"
        else:
            return "mixed"

    def _identify_text_pattern(self, text: str) -> str:
        """Identify pattern in text values."""
        if not isinstance(text, str):
            return "non_text"

        # Check for common patterns
        if re.match(r'^\d{4}-\d{2}-\d{2}', text):
            return "date_iso"
        elif re.match(r'^\d+\.?\d*%?$', text):
            return "numeric_string"
        elif len(text.split()) > 10:
            return "long_text"
        elif len(text.split()) > 3:
            return "sentence"
        elif text.isupper():
            return "uppercase_text"
        else:
            return "short_text"

    def _classify_domain(self, keys_info: Dict[str, Any], values_info: Dict[str, Any]) -> str:
        """Classify domain based on key and value analysis."""
        domain_scores = defaultdict(float)

        # Analyze keys against domain patterns
        all_keys_text = " ".join(keys_info["all_keys"]).lower()
        all_values_text = " ".join(str(v) for v in values_info["text_values"]).lower()
        combined_text = f"{all_keys_text} {all_values_text}"

        for domain, patterns in self.domain_patterns.items():
            score = sum(1 for pattern in patterns if pattern in combined_text)
            domain_scores[domain] = score

        # Add structural scoring
        self._add_structural_domain_scores(keys_info, values_info, domain_scores)

        if not domain_scores or max(domain_scores.values()) == 0:
            return "general"

        return max(domain_scores, key=domain_scores.get)

    def _add_structural_domain_scores(self, keys_info: Dict[str, Any],
                                    values_info: Dict[str, Any],
                                    domain_scores: Dict[str, float]) -> None:
        """Add structural analysis to domain scoring."""

        # Analytics indicators
        if "numeric_string" in values_info["text_patterns"]:
            domain_scores["analytics"] += 2
        if values_info["value_types"]["float"] > values_info["value_types"]["str"]:
            domain_scores["analytics"] += 1

        # Configuration indicators
        if keys_info["key_types"]["identifiers"] > 0:
            domain_scores["configuration"] += 1
        if values_info["value_types"]["bool"] > 0:
            domain_scores["configuration"] += 1

        # Business data indicators
        if keys_info["key_types"]["names"] > 0:
            domain_scores["business_data"] += 1

    def _identify_entity_types(self, keys_info: Dict[str, Any],
                             values_info: Dict[str, Any]) -> Dict[str, int]:
        """Identify entity types based on key and value patterns."""
        entity_types = {}

        for entity_type, count in keys_info["key_types"].items():
            if count > 0:
                entity_types[entity_type] = count

        return entity_types

    def _identify_metric_patterns(self, keys_info: Dict[str, Any],
                                values_info: Dict[str, Any]) -> List[str]:
        """Identify metric patterns in the data."""
        patterns = []

        combined_text = " ".join(keys_info["all_keys"]).lower()

        for pattern_name, pattern_regex in self.metric_patterns.items():
            if re.search(pattern_regex, combined_text, re.IGNORECASE):
                patterns.append(pattern_name)

        # Additional pattern detection based on value types
        if values_info["text_patterns"]["numeric_string"] > 0:
            patterns.append("numeric_strings")
        if len(values_info["numeric_values"]) > len(values_info["text_values"]):
            patterns.append("numeric_heavy")

        return patterns

    def _detect_content_type(self, keys_info: Dict[str, Any],
                           values_info: Dict[str, Any],
                           content_type: str) -> bool:
        """Generic content type detection."""
        if content_type not in self.content_type_patterns:
            return False

        patterns = self.content_type_patterns[content_type]
        combined_text = " ".join(keys_info["all_keys"] +
                               [str(v) for v in values_info["text_values"]]).lower()

        # Check regex patterns
        regex_patterns = [p for p in patterns if p.startswith(r'\\d') or '\\' in p]
        keyword_patterns = [p for p in patterns if p not in regex_patterns]

        # Check keywords
        keyword_match = any(keyword in combined_text for keyword in keyword_patterns)

        # Check regex patterns
        regex_match = any(re.search(pattern, combined_text, re.IGNORECASE)
                         for pattern in regex_patterns)

        return keyword_match or regex_match

    def _detect_performance_metrics(self, keys_info: Dict[str, Any],
                                  values_info: Dict[str, Any]) -> List[str]:
        """Detect performance metrics in the data."""
        performance_keywords = self.config.get("performance_keywords", [
            "performance", "metric", "score", "rate", "change", "growth"
        ])

        combined_text = " ".join(keys_info["all_keys"]).lower()
        found_metrics = [keyword for keyword in performance_keywords if keyword in combined_text]
        return found_metrics

    def _detect_reasoning_content(self, values_info: Dict[str, Any]) -> List[str]:
        """Detect natural language reasoning content."""
        reasoning_keywords = self.config.get("reasoning_keywords", [
            "reason", "because", "due", "analysis", "explanation"
        ])

        long_texts = [v for v in values_info["text_values"]
                     if isinstance(v, str) and len(v.split()) > 5]

        if not long_texts:
            return []

        combined_text = " ".join(long_texts).lower()
        found_reasoning = [keyword for keyword in reasoning_keywords if keyword in combined_text]
        return found_reasoning

    def _identify_metadata_sections(self, data: Dict[str, Any]) -> List[str]:
        """Identify metadata sections generically."""
        metadata_keywords = self.config.get("metadata_keywords", [
            "meta", "config", "info", "header", "settings"
        ])

        sections = []
        for key in data.keys():
            if any(keyword in key.lower() for keyword in metadata_keywords):
                sections.append(key)

        return sections

    def _extract_business_entities(self, data: Dict[str, Any]) -> List[str]:
        """Extract business entity names generically."""
        entities = set()

        # Look for top-level keys that might be entity names
        for key in data.keys():
            if self._is_likely_entity_name(key):
                entities.add(key)

        # Look for entity collections
        entity_collection_keys = self.config.get("entity_collection_keys", [
            "data", "items", "entities", "records", "results"
        ])

        for key in entity_collection_keys:
            if key in data and isinstance(data[key], dict):
                for entity_key in data[key].keys():
                    if self._is_likely_entity_name(entity_key):
                        entities.add(entity_key)

        return list(entities)

    def _is_likely_entity_name(self, name: str) -> bool:
        """Check if a string is likely a business entity name."""
        if not name or len(name) < 2:
            return False

        # Exclude common technical keys
        technical_keys = self.config.get("technical_keys", [
            "id", "type", "count", "total", "sum", "max", "min",
            "config", "meta", "info", "data", "key", "value"
        ])

        return name.lower() not in technical_keys

    def _detect_configuration_data(self, keys_info: Dict[str, Any],
                                 values_info: Dict[str, Any]) -> bool:
        """Detect configuration-style data."""
        config_indicators = keys_info["key_types"].get("identifiers", 0) > 0
        bool_values = values_info["value_types"].get("bool", 0) > 0

        return config_indicators or bool_values

    def _detect_analytics_data(self, keys_info: Dict[str, Any],
                             values_info: Dict[str, Any]) -> bool:
        """Detect analytics-style data."""
        numeric_heavy = len(values_info["numeric_values"]) > len(values_info["text_values"])
        has_metrics = any("metric" in key.lower() for key in keys_info["all_keys"])

        return numeric_heavy or has_metrics