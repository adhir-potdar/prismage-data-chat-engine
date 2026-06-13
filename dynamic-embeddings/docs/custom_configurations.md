# Custom Configuration Guide

The Dynamic JSON Embeddings system supports fully customizable configurations through external JSON files. This allows you to tailor the chunking behavior for your specific domain, data types, and requirements.

## Quick Start

### 1. Basic Usage

```python
from dynamic_embeddings import DynamicChunkingEngine, AnalyzerConfig

# Register a custom configuration
AnalyzerConfig.register_custom_config("my_config", "/path/to/my_config.json")

# Use the custom configuration
engine = DynamicChunkingEngine(config_name="my_config")
chunks, metadata = engine.process_document(your_data)
```

### 2. Load from File Directly

```python
# Load custom config without registration
engine = DynamicChunkingEngine.from_custom_config_file("/path/to/config.json")

# Or load and register in one step
engine = DynamicChunkingEngine.from_custom_config_file(
    "/path/to/config.json",
    register_as="my_config"
)
```

### 3. Import from Directory

```python
# Import all JSON configs from a directory
imported_configs = AnalyzerConfig.import_config_directory(
    "/path/to/configs/",
    prefix="custom"
)

# Use any imported configuration
engine = DynamicChunkingEngine(config_name="custom_medical")
```

## Configuration File Structure

### Basic Structure

```json
{
  "description": "Description of your configuration",
  "version": "1.0",
  "extends": "default.json",

  "domain_patterns": {
    "your_domain": ["keyword1", "keyword2", "keyword3"]
  },

  "metric_patterns": {
    "your_metrics": "pattern1|pattern2|pattern3"
  },

  "performance_keywords": ["performance", "optimization"],
  "reasoning_keywords": ["analysis", "conclusion"]
}
```

### Complete Configuration Options

#### Required Sections

- **domain_patterns**: Domain-specific keyword patterns for content classification
- **metric_patterns**: Regex patterns for identifying metrics and measurements
- **performance_keywords**: Keywords indicating performance-related content
- **reasoning_keywords**: Keywords indicating analytical/reasoning content

#### Optional Sections

- **entity_indicators**: Patterns for identifying different entity types
- **content_type_patterns**: Regex patterns for specific content types (IDs, URLs, etc.)
- **decision_thresholds**: Custom thresholds for strategy selection
- **strategy_weights**: Weights for different factors in decision making
- **chunking_preferences**: Strategy-specific preferences

### Domain Patterns

Define keywords that help identify content domains:

```json
{
  "domain_patterns": {
    "medical": ["patient", "diagnosis", "treatment", "medication"],
    "financial": ["account", "transaction", "balance", "investment"],
    "technical": ["server", "database", "api", "application"],
    "ecommerce": ["product", "order", "cart", "purchase"]
  }
}
```

### Metric Patterns

Regex patterns for identifying metrics and measurements:

```json
{
  "metric_patterns": {
    "percentages": "\\d+%|\\d+\\.\\d+%|percent|percentage",
    "monetary": "\\$\\d+|€\\d+|USD|EUR|revenue|cost|price",
    "performance": "latency|throughput|response_time|cpu_usage",
    "medical_values": "blood_pressure|heart_rate|temperature|glucose"
  }
}
```

### Entity Indicators

Patterns for identifying different types of entities:

```json
{
  "entity_indicators": {
    "persons": ["patient", "customer", "user", "employee"],
    "organizations": ["company", "hospital", "department"],
    "products": ["product", "service", "item", "medication"],
    "locations": ["address", "building", "room", "facility"]
  }
}
```

### Configuration Inheritance

Use the `extends` field to inherit from existing configurations:

```json
{
  "description": "Medical configuration extending business base",
  "extends": "business.json",

  "domain_patterns": {
    "medical": ["patient", "diagnosis", "treatment"]
  }
}
```

Available base configurations:
- `default.json` - Basic patterns
- `analytics.json` - Analytics and metrics
- `advertising.json` - Advertising and ad-tech analytics
- `business.json` - Business entities and processes
- `ecommerce.json` - E-commerce and retail
- `configuration.json` - System configuration data

## Advanced Features

### 1. Configuration Validation

```python
# Validate a configuration file
validation = AnalyzerConfig.validate_custom_config("/path/to/config.json")

if validation['valid']:
    print("Configuration is valid")
else:
    print("Errors:", validation['errors'])
    print("Warnings:", validation['warnings'])
```

### 2. Configuration Templates

```python
# Create a template based on existing configuration
AnalyzerConfig.create_custom_config_template(
    "/path/to/template.json",
    base_config="business"
)
```

### 3. Merging Configurations

```python
# Merge multiple configurations
merged_config = AnalyzerConfig.merge_configs(
    "business",
    "medical",
    output_name="healthcare_business"
)

# Use merged configuration
engine = DynamicChunkingEngine(config_name="healthcare_business")
```

### 4. Dynamic Configuration Switching

```python
# Switch configuration at runtime
engine = DynamicChunkingEngine(config_name="default")

# Switch to custom configuration
engine.switch_config("my_custom_config")

# Switch to built-in configuration
engine.switch_config("ecommerce")
```

### 5. Configuration Management

```python
# List all available configurations
configs = AnalyzerConfig.list_available_configs()
print("Built-in:", configs['built_in'])
print("Custom:", configs['custom'])

# Get custom configuration information
info = AnalyzerConfig.get_custom_config_info()
print(f"Total custom configs: {info['total_custom_configs']}")
print(f"Config details: {info['config_details']}")

# Clear custom configurations
AnalyzerConfig.clear_custom_configs()
```

## Use Cases and Examples

### 1. Medical/Healthcare Data

```json
{
  "description": "Medical and healthcare data configuration",
  "extends": "default.json",

  "domain_patterns": {
    "medical": ["patient", "diagnosis", "treatment", "symptom"],
    "clinical": ["laboratory", "test", "result", "clinical"],
    "pharmaceutical": ["medication", "drug", "dosage", "prescription"]
  },

  "metric_patterns": {
    "vital_signs": "blood_pressure|heart_rate|temperature|oxygen_saturation",
    "lab_values": "hemoglobin|glucose|cholesterol|creatinine"
  },

  "performance_keywords": [
    "efficacy", "effectiveness", "outcome", "recovery", "survival"
  ],

  "reasoning_keywords": [
    "diagnosis", "prognosis", "assessment", "recommendation"
  ]
}
```

### 2. Financial/Trading Data

```json
{
  "description": "Financial and trading data configuration",
  "extends": "business.json",

  "domain_patterns": {
    "trading": ["trade", "stock", "bond", "option", "futures"],
    "banking": ["account", "transaction", "balance", "deposit"],
    "risk": ["risk", "exposure", "volatility", "hedge"]
  },

  "metric_patterns": {
    "monetary": "\\$\\d+|amount|value|price|cost|fee",
    "financial_ratios": "ratio|percentage|yield|return|margin"
  },

  "performance_keywords": [
    "return", "profit", "loss", "performance", "yield"
  ],

  "reasoning_keywords": [
    "analysis", "forecast", "strategy", "recommendation"
  ]
}
```

### 3. IoT/Sensor Data

```json
{
  "description": "IoT and sensor data configuration",

  "domain_patterns": {
    "sensors": ["sensor", "device", "measurement", "reading"],
    "connectivity": ["wifi", "bluetooth", "cellular", "lora"],
    "automation": ["automation", "control", "trigger", "action"]
  },

  "metric_patterns": {
    "sensor_data": "temperature|humidity|pressure|light|motion",
    "connectivity": "signal|strength|quality|bandwidth|rssi"
  },

  "performance_keywords": [
    "efficiency", "accuracy", "reliability", "uptime"
  ],

  "reasoning_keywords": [
    "calibration", "maintenance", "alert", "threshold"
  ]
}
```

## Best Practices

### 1. Configuration Design

- **Start with inheritance**: Extend existing configurations rather than creating from scratch
- **Use descriptive names**: Make domain patterns and metric patterns self-explanatory
- **Include documentation**: Add description and version fields
- **Validate early**: Use validation tools to catch errors

### 2. Domain Patterns

- **Be specific**: Use precise keywords for your domain
- **Include variants**: Add common synonyms and variations
- **Avoid overlap**: Minimize overlap between different domain categories
- **Keep updated**: Regularly review and update patterns based on new data

### 3. Metric Patterns

- **Use regex carefully**: Test patterns thoroughly to avoid false matches
- **Be comprehensive**: Cover all relevant measurement types in your domain
- **Include units**: Capture both numeric values and their units
- **Consider formats**: Account for different number and date formats

### 4. Performance

- **Limit pattern complexity**: Overly complex regex patterns can slow processing
- **Reasonable keyword lists**: Too many keywords can reduce precision
- **Test with real data**: Validate configuration with actual data samples
- **Monitor usage**: Check which patterns are actually being used

### 5. Maintenance

- **Version control**: Keep configuration files in version control
- **Document changes**: Maintain changelog for configuration updates
- **Test changes**: Validate configuration changes with test data
- **Backup configs**: Keep backups of working configurations

## Troubleshooting

### Common Issues

1. **Configuration not found**
   ```
   KeyError: Configuration 'my_config' not found
   ```
   Solution: Ensure configuration is registered or file path is correct

2. **Invalid JSON structure**
   ```
   ValueError: Invalid configuration file
   ```
   Solution: Validate JSON syntax and required fields

3. **Inheritance errors**
   ```
   FileNotFoundError: Parent configuration not found
   ```
   Solution: Ensure parent configuration file exists in config directory

4. **Poor chunking results**
   - Review domain patterns for accuracy
   - Check metric patterns with test data
   - Adjust decision thresholds if needed
   - Consider using different base configuration

### Debugging Tips

- Use configuration validation before registration
- Test with small data samples first
- Enable debug logging to see decision process
- Compare results with built-in configurations
- Use the decision engine recommendations feature

## API Reference

### AnalyzerConfig Methods

- `register_custom_config(name, path)` - Register custom configuration
- `get_custom_config(name)` - Get registered configuration
- `load_custom_config_from_file(path, register_as=None)` - Load from file
- `import_config_directory(directory, prefix=None)` - Import directory
- `validate_custom_config(path)` - Validate configuration
- `create_custom_config_template(output_path, base_config)` - Create template
- `merge_configs(*names, output_name=None)` - Merge configurations
- `list_available_configs()` - List all configurations
- `clear_custom_configs()` - Clear custom registry

### DynamicChunkingEngine Methods

- `from_custom_config_file(path, register_as=None)` - Create from file
- `from_config_name(name)` - Create from configuration name
- `switch_config(name)` - Switch to different configuration
- `load_custom_configs_from_directory(directory, prefix=None)` - Load from directory

For complete API documentation, see the source code docstrings.