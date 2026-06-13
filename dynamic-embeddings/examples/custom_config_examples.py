"""Examples demonstrating custom configuration capabilities."""

import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from dynamic_embeddings import DynamicChunkingEngine, AnalyzerConfig


def demonstrate_custom_config_registration():
    """Demonstrate registering and using custom configurations."""

    print("=== Custom Configuration Registration Demo ===\n")

    # Create a custom configuration
    custom_config_path = Path(__file__).parent / "custom_medical_config.json"

    medical_config = {
        "description": "Configuration for medical and healthcare data analysis",
        "version": "1.0",
        "extends": "default.json",
        "domain_patterns": {
            "medical": ["patient", "diagnosis", "treatment", "medication", "symptom"],
            "clinical": ["clinical", "laboratory", "test", "result", "analysis"],
            "healthcare": ["healthcare", "hospital", "clinic", "doctor", "nurse"],
            "pharmaceutical": ["drug", "dosage", "prescription", "pharmacy", "medicine"]
        },
        "metric_patterns": {
            "vital_signs": "blood_pressure|heart_rate|temperature|oxygen_saturation",
            "lab_values": "hemoglobin|glucose|cholesterol|creatinine",
            "measurements": "weight|height|bmi|age"
        },
        "entity_indicators": {
            "patients": ["patient", "subject", "individual", "case"],
            "medical_staff": ["doctor", "physician", "nurse", "practitioner"],
            "facilities": ["hospital", "clinic", "laboratory", "pharmacy"],
            "procedures": ["surgery", "procedure", "intervention", "therapy"]
        },
        "performance_keywords": [
            "efficacy", "effectiveness", "outcome", "improvement", "recovery",
            "response", "survival", "mortality", "morbidity"
        ],
        "reasoning_keywords": [
            "diagnosis", "prognosis", "assessment", "evaluation", "recommendation",
            "contraindication", "indication", "adverse_effect", "side_effect"
        ]
    }

    # Save custom config
    with open(custom_config_path, 'w') as f:
        json.dump(medical_config, f, indent=2)

    print(f"1. Created custom medical configuration: {custom_config_path}")

    # Register the custom configuration
    AnalyzerConfig.register_custom_config("medical", str(custom_config_path))
    print("2. Registered custom configuration as 'medical'")

    # List available configurations
    available = AnalyzerConfig.list_available_configs()
    print(f"3. Available configurations:")
    print(f"   Built-in: {available['built_in']}")
    print(f"   Custom: {available['custom']}")
    print()

    # Test with medical data
    medical_data = {
        "patient_record": {
            "patient_id": "PAT001",
            "demographics": {
                "age": 45,
                "gender": "female",
                "weight": 70.2,
                "height": 165
            },
            "vital_signs": {
                "blood_pressure": "120/80",
                "heart_rate": 72,
                "temperature": 36.8,
                "oxygen_saturation": "98%"
            },
            "laboratory_results": {
                "hemoglobin": 12.5,
                "glucose": 95,
                "cholesterol": 180,
                "reasoning": "All laboratory values within normal range, indicating good metabolic health"
            },
            "diagnosis": {
                "primary": "Hypertension",
                "secondary": ["Type 2 Diabetes", "Obesity"],
                "clinical_assessment": "Patient shows good response to current treatment regimen with stable vital signs and improving lab values"
            }
        }
    }

    # Use custom configuration
    engine = DynamicChunkingEngine(config_name="medical")
    chunks, metadata = engine.process_document(medical_data, "patient_001")

    print(f"4. Processed medical data using custom config:")
    print(f"   Strategy used: {metadata['strategy_used']}")
    print(f"   Chunks created: {len(chunks)}")
    print(f"   Domain detected: {metadata['decision_details']['content_analysis']['domain_type']}")

    # Show chunk details
    for i, chunk in enumerate(chunks[:2]):
        print(f"   Chunk {i+1}: {chunk.metadata.chunk_type}")
        medical_tags = [tag for tag in chunk.metadata.domain_tags if 'medical' in tag.lower()]
        if medical_tags:
            print(f"     Medical tags: {medical_tags}")

    print()


def demonstrate_config_directory_import():
    """Demonstrate importing configurations from a directory."""

    print("=== Directory Import Demo ===\n")

    # Create a custom configs directory
    custom_configs_dir = Path(__file__).parent / "custom_configs"
    custom_configs_dir.mkdir(exist_ok=True)

    # Create multiple custom configurations
    configs_to_create = {
        "financial.json": {
            "description": "Financial and banking data configuration",
            "domain_patterns": {
                "banking": ["account", "transaction", "balance", "deposit", "withdrawal"],
                "finance": ["finance", "investment", "portfolio", "asset", "liability"],
                "trading": ["trade", "stock", "bond", "option", "futures"],
                "risk": ["risk", "exposure", "volatility", "hedge", "derivative"]
            },
            "metric_patterns": {
                "monetary": "amount|value|price|cost|fee|interest|rate",
                "financial_ratios": "ratio|percentage|yield|return|margin"
            },
            "performance_keywords": ["return", "profit", "loss", "performance", "yield"]
        },

        "iot.json": {
            "description": "IoT and sensor data configuration",
            "domain_patterns": {
                "sensors": ["sensor", "device", "measurement", "reading", "data"],
                "connectivity": ["wifi", "bluetooth", "cellular", "lora", "zigbee"],
                "automation": ["automation", "control", "trigger", "action", "rule"]
            },
            "metric_patterns": {
                "sensor_data": "temperature|humidity|pressure|light|motion",
                "connectivity": "signal|strength|quality|bandwidth|latency"
            },
            "performance_keywords": ["efficiency", "accuracy", "reliability", "uptime"]
        },

        "gaming.json": {
            "description": "Gaming and entertainment data configuration",
            "domain_patterns": {
                "gaming": ["game", "player", "level", "score", "achievement"],
                "multiplayer": ["guild", "team", "match", "tournament", "ranking"],
                "monetization": ["purchase", "microtransaction", "subscription", "dlc"]
            },
            "metric_patterns": {
                "game_metrics": "score|points|level|experience|rank",
                "engagement": "session|duration|retention|churn"
            },
            "performance_keywords": ["engagement", "retention", "monetization", "performance"]
        }
    }

    # Create the config files
    for filename, config in configs_to_create.items():
        config_path = custom_configs_dir / filename
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    print(f"1. Created {len(configs_to_create)} custom configuration files in {custom_configs_dir}")

    # Import all configurations from directory
    imported_configs = AnalyzerConfig.import_config_directory(str(custom_configs_dir), prefix="custom")

    print(f"2. Imported configurations: {imported_configs}")

    # Test each configuration
    test_data = {
        "metrics": {"value": 100, "performance": "excellent"},
        "analysis": "System performing within expected parameters"
    }

    for config_name in imported_configs:
        engine = DynamicChunkingEngine(config_name=config_name)
        strategy, details = engine.decision_engine.decide_strategy(test_data)
        print(f"   {config_name}: strategy={strategy.value}, domain={details['content_analysis']['domain_type']}")

    print()


def demonstrate_config_validation_and_templates():
    """Demonstrate configuration validation and template creation."""

    print("=== Configuration Validation and Templates Demo ===\n")

    # Create a template
    template_path = Path(__file__).parent / "custom_template.json"
    AnalyzerConfig.create_custom_config_template(str(template_path), "business")
    print(f"1. Created configuration template: {template_path}")

    # Validate the template
    validation = AnalyzerConfig.validate_custom_config(str(template_path))
    print(f"2. Template validation:")
    print(f"   Valid: {validation['valid']}")
    if validation['warnings']:
        print(f"   Warnings: {validation['warnings']}")
    if validation['suggestions']:
        print(f"   Suggestions: {validation['suggestions']}")

    # Create an invalid configuration for testing
    invalid_config_path = Path(__file__).parent / "invalid_config.json"
    invalid_config = {
        "domain_patterns": "this_should_be_a_dict",  # Invalid structure
        "metric_patterns": ["this", "should", "be", "dict"]  # Invalid structure
    }

    with open(invalid_config_path, 'w') as f:
        json.dump(invalid_config, f, indent=2)

    # Validate the invalid configuration
    print(f"\n3. Testing invalid configuration:")
    validation = AnalyzerConfig.validate_custom_config(str(invalid_config_path))
    print(f"   Valid: {validation['valid']}")
    if validation['errors']:
        print(f"   Errors: {validation['errors']}")

    print()


def demonstrate_config_merging():
    """Demonstrate merging multiple configurations."""

    print("=== Configuration Merging Demo ===\n")

    # Register some custom configs first (from previous examples)
    if "medical" not in AnalyzerConfig.list_available_configs()['custom']:
        print("Skipping merge demo - custom configs not available")
        return

    # Merge configurations
    merged_config = AnalyzerConfig.merge_configs("business", "medical", output_name="healthcare_business")

    print("1. Merged 'business' and 'medical' configurations into 'healthcare_business'")

    # Show merged configuration details
    print(f"2. Merged configuration contains:")
    print(f"   Domain patterns: {len(merged_config.get('domain_patterns', {}))}")
    print(f"   Metric patterns: {len(merged_config.get('metric_patterns', {}))}")
    print(f"   Merged from: {merged_config.get('_merge_info', {}).get('merged_from', [])}")

    # Test merged configuration
    test_data = {
        "hospital_operations": {
            "patient_statistics": {"total_patients": 150, "average_stay": 3.5},
            "financial_metrics": {"revenue": 500000, "costs": 350000},
            "clinical_outcomes": {
                "patient_satisfaction": "95%",
                "readmission_rate": "8%",
                "reasoning": "Improved patient outcomes due to enhanced care protocols"
            }
        }
    }

    engine = DynamicChunkingEngine(config_name="healthcare_business")
    chunks, metadata = engine.process_document(test_data, "hospital_data")

    print(f"3. Processed healthcare business data:")
    print(f"   Strategy: {metadata['strategy_used']}")
    print(f"   Domain detected: {metadata['decision_details']['content_analysis']['domain_type']}")
    print(f"   Chunks created: {len(chunks)}")

    print()


def demonstrate_engine_config_methods():
    """Demonstrate engine-specific configuration methods."""

    print("=== Engine Configuration Methods Demo ===\n")

    # Create engine from custom config file
    custom_config_path = Path(__file__).parent / "custom_medical_config.json"
    if custom_config_path.exists():
        engine1 = DynamicChunkingEngine.from_custom_config_file(str(custom_config_path), "temp_medical")
        print("1. Created engine from custom config file")

        # Get engine info
        info = engine1.get_engine_info()
        print(f"   Custom configs available: {len(info['custom_config_info']['config_names'])}")
        print(f"   Config domains: {info['config_domains']}")

    # Create engine from config name
    engine2 = DynamicChunkingEngine.from_config_name("default")
    print("2. Created engine from config name 'default'")

    # Switch configuration
    if "medical" in AnalyzerConfig.list_available_configs()['custom']:
        engine2.switch_config("medical")
        print("3. Switched engine to 'medical' configuration")

    # Show custom config info
    custom_info = AnalyzerConfig.get_custom_config_info()
    print(f"4. Custom configuration summary:")
    print(f"   Total custom configs: {custom_info['total_custom_configs']}")
    for name, details in custom_info['config_details'].items():
        print(f"   {name}: {details['domain_patterns']} domains, inheritance: {details['has_inheritance']}")

    print()


def cleanup_demo_files():
    """Clean up demo files created during examples."""

    print("=== Cleanup ===\n")

    files_to_remove = [
        "custom_medical_config.json",
        "custom_template.json",
        "invalid_config.json"
    ]

    directories_to_remove = [
        "custom_configs"
    ]

    # Remove files
    for filename in files_to_remove:
        file_path = Path(__file__).parent / filename
        if file_path.exists():
            file_path.unlink()
            print(f"Removed: {filename}")

    # Remove directories
    for dirname in directories_to_remove:
        dir_path = Path(__file__).parent / dirname
        if dir_path.exists():
            for file in dir_path.glob("*"):
                file.unlink()
            dir_path.rmdir()
            print(f"Removed directory: {dirname}")

    # Clear custom configurations from registry
    AnalyzerConfig.clear_custom_configs()
    print("Cleared custom configuration registry")

    print("Cleanup completed!")


if __name__ == "__main__":
    try:
        demonstrate_custom_config_registration()
        demonstrate_config_directory_import()
        demonstrate_config_validation_and_templates()
        demonstrate_config_merging()
        demonstrate_engine_config_methods()
    finally:
        cleanup_demo_files()