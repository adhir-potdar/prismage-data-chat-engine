"""Configuration for analyzers with customizable patterns and rules."""

from typing import Any, Dict, List, Optional
import json
from pathlib import Path
import os


class AnalyzerConfig:
    """Configuration class for content and structure analyzers."""

    # Get the config directory relative to this file
    _CONFIG_DIR = Path(__file__).parent.parent.parent.parent / "config"

    # Registry for custom configurations
    _custom_configs = {}
    _custom_config_paths = []

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Load default configuration from external file."""
        return cls.load_from_file(cls._CONFIG_DIR / "default.json")

    @classmethod
    def get_analytics_config(cls) -> Dict[str, Any]:
        """Load analytics configuration from external file."""
        return cls.load_from_file(cls._CONFIG_DIR / "analytics.json")

    @classmethod
    def get_advertising_config(cls) -> Dict[str, Any]:
        """Load advertising/ad-tech analytics configuration from external file."""
        return cls.load_from_file(cls._CONFIG_DIR / "advertising.json")

    @classmethod
    def get_configuration_config(cls) -> Dict[str, Any]:
        """Load configuration files analysis config from external file."""
        return cls.load_from_file(cls._CONFIG_DIR / "configuration.json")

    @classmethod
    def get_business_config(cls) -> Dict[str, Any]:
        """Load business data configuration from external file."""
        return cls.load_from_file(cls._CONFIG_DIR / "business.json")

    @classmethod
    def get_ecommerce_config(cls) -> Dict[str, Any]:
        """Load e-commerce configuration from external file."""
        return cls.load_from_file(cls._CONFIG_DIR / "ecommerce.json")

    @classmethod
    def get_minimal_config(cls) -> Dict[str, Any]:
        """Get minimal configuration with required fields."""
        return {
            "description": "Minimal configuration",
            "version": "1.0",
            "domain_patterns": {
                "general": ["data", "info", "content", "item"]
            },
            "metric_patterns": {
                "basic": "count|total|number|value"
            },
            "entity_indicators": {
                "basic": ["id", "name", "type", "key"]
            },
            "content_type_patterns": {},
            "performance_keywords": ["performance", "metric", "score"],
            "reasoning_keywords": ["reason", "analysis", "explanation"],
            "decision_thresholds": {
                "max_flat_depth": 3,
                "max_flat_keys": 20,
                "min_hierarchical_depth": 4,
                "semantic_content_ratio": 0.3,
                "dimensional_array_ratio": 0.4,
                "large_document_size": 1000000
            },
            "strategy_weights": {
                "structure_weight": 0.4,
                "content_weight": 0.3,
                "size_weight": 0.2,
                "performance_weight": 0.1
            }
        }

    @classmethod
    def load_from_file(cls, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON or YAML file with inheritance support."""
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(path, 'r') as f:
            if path.suffix.lower() == '.json':
                config = json.load(f)
            elif path.suffix.lower() in ['.yml', '.yaml']:
                try:
                    import yaml
                    config = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML is required for YAML configuration files")
            else:
                raise ValueError("Configuration file must be JSON or YAML")

        # Handle inheritance (extends)
        if "extends" in config:
            parent_path = cls._CONFIG_DIR / config["extends"]
            parent_config = cls.load_from_file(str(parent_path))

            # Merge parent config with current config
            merged_config = cls._deep_merge(parent_config, config)
            return merged_config

        return config

    @classmethod
    def _deep_merge(cls, base_dict: Dict[str, Any], override_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries, with override_dict taking precedence."""
        result = base_dict.copy()

        for key, value in override_dict.items():
            if key == "extends":
                # Skip the extends key in final config
                continue
            elif key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = cls._deep_merge(result[key], value)
            elif key in result and isinstance(result[key], list) and isinstance(value, list):
                # For lists, extend rather than replace
                result[key] = result[key] + value
            else:
                result[key] = value

        return result

    @classmethod
    def create_config(
        cls,
        base_config: Optional[Dict[str, Any]] = None,
        domain_patterns: Optional[Dict[str, List[str]]] = None,
        metric_patterns: Optional[Dict[str, str]] = None,
        entity_indicators: Optional[Dict[str, List[str]]] = None,
        content_type_patterns: Optional[Dict[str, List[str]]] = None,
        keywords: Optional[Dict[str, List[str]]] = None,
        **additional_config
    ) -> Dict[str, Any]:
        """Create configuration by combining base config with custom patterns."""

        # Start with minimal config if no base provided
        config = base_config if base_config else cls.get_minimal_config()

        # Update domain patterns
        if domain_patterns:
            config.setdefault("domain_patterns", {})
            config["domain_patterns"].update(domain_patterns)

        # Update metric patterns
        if metric_patterns:
            config.setdefault("metric_patterns", {})
            config["metric_patterns"].update(metric_patterns)

        # Update entity indicators
        if entity_indicators:
            config.setdefault("entity_indicators", {})
            config["entity_indicators"].update(entity_indicators)

        # Update content type patterns
        if content_type_patterns:
            config.setdefault("content_type_patterns", {})
            config["content_type_patterns"].update(content_type_patterns)

        # Update keyword lists
        if keywords:
            for key_type, keyword_list in keywords.items():
                keyword_key = f"{key_type}_keywords"
                if keyword_key in config:
                    config[keyword_key].extend(keyword_list)
                else:
                    config[keyword_key] = keyword_list

        # Update any additional configuration
        for key, value in additional_config.items():
            if key in config:
                if isinstance(config[key], dict) and isinstance(value, dict):
                    config[key].update(value)
                elif isinstance(config[key], list) and isinstance(value, list):
                    config[key].extend(value)
                else:
                    config[key] = value
            else:
                config[key] = value

        return config

    @classmethod
    def from_json_sample(cls, json_data: Dict[str, Any], config_hint: Optional[str] = None) -> Dict[str, Any]:
        """Generate configuration by analyzing a sample JSON structure."""
        config = cls.get_minimal_config()

        # Extract patterns from the JSON data
        discovered_patterns = cls._discover_patterns_from_json(json_data)

        # Merge discovered patterns with base config
        for pattern_type, patterns in discovered_patterns.items():
            if pattern_type in config:
                if isinstance(config[pattern_type], dict):
                    config[pattern_type].update(patterns)
                elif isinstance(config[pattern_type], list):
                    config[pattern_type].extend(patterns)

        return config

    @staticmethod
    def _discover_patterns_from_json(json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Discover patterns from a JSON data sample."""
        patterns = {
            "domain_patterns": {"discovered": []},
            "entity_indicators": {"discovered_keys": []},
            "performance_keywords": [],
            "reasoning_keywords": []
        }

        def collect_keys_and_values(obj, keys_list, values_list):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    keys_list.append(key)
                    if isinstance(value, str):
                        values_list.append(value)
                    collect_keys_and_values(value, keys_list, values_list)
            elif isinstance(obj, list):
                for item in obj:
                    collect_keys_and_values(item, keys_list, values_list)

        keys_list = []
        values_list = []
        collect_keys_and_values(json_data, keys_list, values_list)

        # Discover common key patterns
        unique_keys = list(set(keys_list))
        patterns["entity_indicators"]["discovered_keys"] = unique_keys[:10]  # Limit to prevent bloat

        # Discover domain-specific terms from keys
        patterns["domain_patterns"]["discovered"] = [
            key.lower() for key in unique_keys
            if len(key) > 2 and not key.isnumeric()
        ][:20]  # Limit to prevent bloat

        return patterns

    @staticmethod
    def save_config(config: Dict[str, Any], output_path: str) -> None:
        """Save configuration to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def create_example_configs(cls) -> Dict[str, Dict[str, Any]]:
        """Create example configurations for different use cases."""
        examples = {}

        # Example 1: Basic analytics configuration
        examples["analytics"] = cls.create_config(
            domain_patterns={
                "analytics": ["metric", "analysis", "performance", "trend"]
            },
            metric_patterns={
                "percentage": r"percent|%|ratio",
                "change": r"change|delta|growth"
            },
            keywords={
                "performance": ["performance", "metric", "kpi"],
                "reasoning": ["analysis", "insight", "conclusion"]
            }
        )

        # Example 2: Configuration file analysis
        examples["configuration"] = cls.create_config(
            domain_patterns={
                "configuration": ["config", "setting", "option", "parameter"]
            },
            entity_indicators={
                "settings": ["setting", "option", "preference"],
                "flags": ["enabled", "disabled", "flag", "toggle"]
            }
        )

        # Example 3: Business data configuration
        examples["business"] = cls.create_config(
            domain_patterns={
                "business": ["customer", "product", "order", "user"]
            },
            entity_indicators={
                "entities": ["customer", "product", "order"],
                "attributes": ["name", "id", "status", "type"]
            }
        )

        return examples

    @classmethod
    def register_custom_config(cls, config_name: str, config_path: str) -> None:
        """Register a custom configuration file.

        Args:
            config_name: Name to register the configuration under
            config_path: Path to the custom configuration file
        """
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Custom configuration file not found: {config_path}")

        # Validate the configuration file
        try:
            config = cls.load_from_file(str(path))
            cls._custom_configs[config_name] = config
            cls._custom_config_paths.append(str(path))
            print(f"Successfully registered custom configuration: {config_name}")
        except Exception as e:
            raise ValueError(f"Invalid configuration file {config_path}: {str(e)}")

    @classmethod
    def get_custom_config(cls, config_name: str) -> Dict[str, Any]:
        """Get a registered custom configuration.

        Args:
            config_name: Name of the custom configuration

        Returns:
            Configuration dictionary

        Raises:
            KeyError: If configuration name is not registered
        """
        if config_name not in cls._custom_configs:
            raise KeyError(f"Custom configuration '{config_name}' not found. "
                          f"Available: {list(cls._custom_configs.keys())}")

        return cls._custom_configs[config_name].copy()

    @classmethod
    def load_custom_config_from_file(cls, config_path: str, register_as: Optional[str] = None) -> Dict[str, Any]:
        """Load custom configuration directly from file.

        Args:
            config_path: Path to the custom configuration file
            register_as: Optional name to register this config for future use

        Returns:
            Configuration dictionary
        """
        config = cls.load_from_file(config_path)

        if register_as:
            cls._custom_configs[register_as] = config
            cls._custom_config_paths.append(config_path)

        return config

    @classmethod
    def import_config_directory(cls, directory_path: str, prefix: Optional[str] = None) -> List[str]:
        """Import all JSON configuration files from a directory.

        Args:
            directory_path: Path to directory containing config files
            prefix: Optional prefix to add to configuration names

        Returns:
            List of imported configuration names
        """
        directory = Path(directory_path)

        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Invalid directory: {directory_path}")

        imported_configs = []
        config_files = list(directory.glob("*.json"))

        for config_file in config_files:
            config_name = config_file.stem
            if prefix:
                config_name = f"{prefix}_{config_name}"

            try:
                cls.register_custom_config(config_name, str(config_file))
                imported_configs.append(config_name)
            except Exception as e:
                print(f"Failed to import {config_file.name}: {str(e)}")

        print(f"Imported {len(imported_configs)} configurations from {directory_path}")
        return imported_configs

    @classmethod
    def list_available_configs(cls) -> Dict[str, List[str]]:
        """List all available configurations.

        Returns:
            Dictionary with built-in and custom configuration names
        """
        built_in_configs = [
            'default', 'analytics', 'advertising', 'configuration',
            'business', 'ecommerce'
        ]

        return {
            'built_in': built_in_configs,
            'custom': list(cls._custom_configs.keys())
        }

    @classmethod
    def get_config_by_name(cls, config_name: str) -> Dict[str, Any]:
        """Get configuration by name (built-in or custom).

        Args:
            config_name: Name of the configuration

        Returns:
            Configuration dictionary
        """
        # Try built-in configurations first
        built_in_methods = {
            'default': cls.get_default_config,
            'analytics': cls.get_analytics_config,
            'advertising': cls.get_advertising_config,
            'configuration': cls.get_configuration_config,
            'business': cls.get_business_config,
            'ecommerce': cls.get_ecommerce_config
        }

        if config_name in built_in_methods:
            return built_in_methods[config_name]()

        # Try custom configurations
        if config_name in cls._custom_configs:
            return cls.get_custom_config(config_name)

        # If not found, suggest available options
        available = cls.list_available_configs()
        all_configs = available['built_in'] + available['custom']
        raise KeyError(f"Configuration '{config_name}' not found. Available: {all_configs}")

    @classmethod
    def create_custom_config_template(cls, output_path: str, base_config: str = "default") -> None:
        """Create a template for custom configuration.

        Args:
            output_path: Path where to save the template
            base_config: Base configuration to use as template
        """
        template = cls.get_config_by_name(base_config)

        # Add template metadata
        template["_template_info"] = {
            "description": "Custom configuration template",
            "base_config": base_config,
            "created_by": "dynamic_embeddings",
            "version": "1.0.0"
        }

        # Add example customizations
        template["_customization_examples"] = {
            "domain_patterns": {
                "my_domain": ["keyword1", "keyword2", "keyword3"]
            },
            "metric_patterns": {
                "my_metrics": "pattern1|pattern2|pattern3"
            },
            "performance_keywords": ["performance_term1", "performance_term2"],
            "reasoning_keywords": ["reasoning_term1", "reasoning_term2"]
        }

        cls.save_config(template, output_path)
        print(f"Custom configuration template saved to: {output_path}")

    @classmethod
    def validate_custom_config(cls, config_path: str) -> Dict[str, Any]:
        """Validate a custom configuration file.

        Args:
            config_path: Path to configuration file to validate

        Returns:
            Validation results
        """
        validation_result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "suggestions": []
        }

        try:
            config = cls.load_from_file(config_path)

            # Required sections check
            required_sections = ["domain_patterns"]
            for section in required_sections:
                if section not in config:
                    validation_result["warnings"].append(f"Missing recommended section: {section}")

            # Validate domain_patterns structure
            if "domain_patterns" in config:
                domain_patterns = config["domain_patterns"]
                if not isinstance(domain_patterns, dict):
                    validation_result["errors"].append("domain_patterns must be a dictionary")
                    validation_result["valid"] = False
                else:
                    for domain, patterns in domain_patterns.items():
                        if not isinstance(patterns, list):
                            validation_result["errors"].append(f"Patterns for domain '{domain}' must be a list")
                            validation_result["valid"] = False
                        elif len(patterns) == 0:
                            validation_result["warnings"].append(f"Domain '{domain}' has no patterns")

            # Validate metric_patterns if present
            if "metric_patterns" in config:
                metric_patterns = config["metric_patterns"]
                if not isinstance(metric_patterns, dict):
                    validation_result["errors"].append("metric_patterns must be a dictionary")
                    validation_result["valid"] = False

            # Check for inheritance
            if "extends" in config:
                parent_file = config["extends"]
                parent_path = Path(config_path).parent / parent_file
                if not parent_path.exists():
                    validation_result["errors"].append(f"Parent configuration not found: {parent_file}")
                    validation_result["valid"] = False

            # Suggestions
            if len(config.get("domain_patterns", {})) > 20:
                validation_result["suggestions"].append("Consider splitting large configurations into multiple files")

            if "description" not in config:
                validation_result["suggestions"].append("Add a 'description' field to document the configuration purpose")

        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Configuration loading error: {str(e)}")

        return validation_result

    @classmethod
    def merge_configs(cls, *config_names: str, output_name: Optional[str] = None) -> Dict[str, Any]:
        """Merge multiple configurations into one.

        Args:
            *config_names: Names of configurations to merge
            output_name: Optional name to register the merged config

        Returns:
            Merged configuration dictionary
        """
        if not config_names:
            raise ValueError("At least one configuration name must be provided")

        base_config = cls.get_config_by_name(config_names[0])

        for config_name in config_names[1:]:
            config_to_merge = cls.get_config_by_name(config_name)
            base_config = cls._deep_merge(base_config, config_to_merge)

        # Add merge metadata
        base_config["_merge_info"] = {
            "merged_from": list(config_names),
            "merge_order": list(config_names)
        }

        if output_name:
            cls._custom_configs[output_name] = base_config

        return base_config

    @classmethod
    def clear_custom_configs(cls) -> None:
        """Clear all registered custom configurations."""
        cls._custom_configs.clear()
        cls._custom_config_paths.clear()
        print("All custom configurations cleared")

    @classmethod
    def get_custom_config_info(cls) -> Dict[str, Any]:
        """Get information about registered custom configurations."""
        return {
            "total_custom_configs": len(cls._custom_configs),
            "config_names": list(cls._custom_configs.keys()),
            "config_paths": cls._custom_config_paths.copy(),
            "config_details": {
                name: {
                    "domain_patterns": len(config.get("domain_patterns", {})),
                    "metric_patterns": len(config.get("metric_patterns", {})),
                    "has_inheritance": "extends" in config
                }
                for name, config in cls._custom_configs.items()
            }
        }