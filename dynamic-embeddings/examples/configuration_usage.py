"""Example usage of external configuration files for different domains."""

import json
from pathlib import Path
from typing import Dict, Any

# Assuming we're running from the examples directory
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from dynamic_embeddings.config.analyzer_config import AnalyzerConfig
from dynamic_embeddings.analyzers.content import ContentAnalyzer


def demonstrate_configuration_usage():
    """Demonstrate how to use external configuration files."""

    print("=== Dynamic Embeddings Configuration Usage Examples ===\n")

    # Example 1: Default configuration
    print("1. Default Configuration:")
    default_config = AnalyzerConfig.get_default_config()
    print(f"   Domain patterns: {list(default_config['domain_patterns'].keys())}")
    print(f"   Metric patterns: {list(default_config['metric_patterns'].keys())}")
    print()

    # Example 2: Analytics configuration (extends default)
    print("2. Analytics Configuration:")
    analytics_config = AnalyzerConfig.get_analytics_config()
    print(f"   Domain patterns: {list(analytics_config['domain_patterns'].keys())}")
    print(f"   Performance keywords: {analytics_config['performance_keywords'][:5]}...")
    print()

    # Example 3: Advertising configuration (extends analytics)
    print("3. Advertising/Ad-Tech Analytics Configuration:")
    advertising_config = AnalyzerConfig.get_advertising_config()
    print(f"   Domain patterns: {list(advertising_config['domain_patterns'].keys())}")
    print(f"   Advertising metrics: {advertising_config['metric_patterns']['advertising_metrics']}")
    print()

    # Example 4: Using configurations with analyzers
    print("4. Using Configurations with Analyzers:")

    # Sample ad-tech analytics data
    sample_adtech_data = {
        "analysis_metadata": {
            "period1": {"start_date": "2024-01-01", "end_date": "2024-01-02"},
            "period2": {"start_date": "2024-01-03", "end_date": "2024-01-04"}
        },
        "hierarchical_analysis": {
            "Property A": {
                "period1_ecpm": 10.50,
                "period2_ecpm": 12.30,
                "change_percentage": "17.14",
                "reasoning": "eCPM increased due to higher fill rates and optimized inventory"
            }
        }
    }

    # Analyze with advertising config
    analyzer = ContentAnalyzer(advertising_config)
    result = analyzer.analyze(sample_adtech_data)

    print(f"   Detected domain: {result.domain_type}")
    print(f"   Performance metrics detected: {result.performance_metrics}")
    print(f"   Reasoning content detected: {result.reasoning_content}")
    print(f"   Entity types: {result.entity_types}")
    print()

    # Example 5: Custom configuration
    print("5. Custom Configuration Creation:")
    custom_config = AnalyzerConfig.create_config(
        base_config=AnalyzerConfig.get_default_config(),
        domain_patterns={
            "my_custom_domain": ["custom_keyword1", "custom_keyword2"]
        },
        metric_patterns={
            "custom_metric": r"my_custom_pattern|another_pattern"
        },
        keywords={
            "performance": ["my_custom_performance_indicator"],
            "reasoning": ["my_custom_reasoning_word"]
        }
    )
    print(f"   Custom domain patterns: {custom_config['domain_patterns']['my_custom_domain']}")
    print()

    # Example 6: Loading custom config from file
    print("6. Loading Custom Configuration from File:")
    custom_config_path = Path(__file__).parent.parent / "config" / "ecommerce.json"
    if custom_config_path.exists():
        ecommerce_config = AnalyzerConfig.load_from_file(str(custom_config_path))
        print(f"   E-commerce domain patterns: {list(ecommerce_config['domain_patterns'].keys())}")
        print(f"   E-commerce metrics: {ecommerce_config['metric_patterns']['ecommerce_metrics']}")
    print()


def demonstrate_config_inheritance():
    """Demonstrate configuration inheritance."""
    print("=== Configuration Inheritance Example ===\n")

    # Load advertising config which extends analytics which extends default
    advertising_config = AnalyzerConfig.get_advertising_config()

    print("Advertising config inherits from:")
    print("  default.json → analytics.json → advertising.json")
    print()

    print("Final merged configuration contains:")
    print(f"- Domain patterns from all levels: {len(advertising_config['domain_patterns'])} domains")
    print(f"- Metric patterns from all levels: {len(advertising_config['metric_patterns'])} patterns")
    print(f"- Entity indicators from all levels: {len(advertising_config['entity_indicators'])} types")
    print()

    # Show specific inheritance examples
    print("Inherited elements:")
    print(f"- From default: {advertising_config['domain_patterns']['general']}")
    print(f"- From analytics: {advertising_config['domain_patterns']['analytics']}")
    print(f"- From advertising: {advertising_config['domain_patterns']['advertising']}")
    print()


def create_custom_domain_config():
    """Example of creating a completely custom domain configuration."""
    print("=== Creating Custom Domain Configuration ===\n")

    # Create config for IoT sensor data
    iot_config = AnalyzerConfig.create_config(
        domain_patterns={
            "iot": ["sensor", "device", "telemetry", "measurement", "iot"],
            "sensors": ["temperature", "humidity", "pressure", "motion", "light"],
            "connectivity": ["wifi", "bluetooth", "cellular", "lora", "zigbee"],
            "data_types": ["reading", "measurement", "value", "signal", "status"]
        },
        metric_patterns={
            "sensor_readings": r"temperature|humidity|pressure|voltage",
            "signal_quality": r"rssi|snr|signal_strength|battery",
            "timestamps": r"timestamp|datetime|recorded_at"
        },
        entity_indicators={
            "devices": ["device", "sensor", "gateway", "node"],
            "locations": ["room", "building", "floor", "zone", "area"],
            "measurements": ["reading", "value", "measurement", "data"]
        },
        keywords={
            "performance": ["signal_quality", "battery_level", "connectivity"],
            "reasoning": ["calibration", "maintenance", "alert", "threshold"]
        }
    )

    print("Created IoT configuration:")
    print(f"- IoT domain patterns: {iot_config['domain_patterns']['iot']}")
    print(f"- Sensor patterns: {iot_config['domain_patterns']['sensors']}")
    print(f"- Sensor reading metrics: {iot_config['metric_patterns']['sensor_readings']}")
    print()

    # Save the custom config
    output_path = Path(__file__).parent.parent / "config" / "iot.json"
    AnalyzerConfig.save_config(iot_config, str(output_path))
    print(f"Saved custom IoT config to: {output_path}")
    print()


if __name__ == "__main__":
    demonstrate_configuration_usage()
    demonstrate_config_inheritance()
    create_custom_domain_config()