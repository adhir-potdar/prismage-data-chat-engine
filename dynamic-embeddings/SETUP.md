# Dynamic JSON Embeddings - Setup Guide

## Quick Setup

### 1. Create Virtual Environment
```bash
cd /Users/adhirpotdar/Work/git-repos/dynamic-embeddings
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -e .
```

### 3. Run Examples
```bash
# Use the example runner script
./run_examples.sh

# Or run examples individually
python examples/decision_engine_usage.py
python examples/complete_system_demo.py
python examples/custom_config_examples.py
python examples/configuration_usage.py
```

## Example Output

When you run the examples, you should see output like:

### Decision Engine Usage
```
=== Dynamic Chunking Decision Engine Examples ===

1. Simple Flat Document:
   Recommended Strategy: flat
   Confidence: 0.99
   Reasoning: Document is relatively simple...

2. Complex Hierarchical Document:
   Recommended Strategy: hierarchical
   Confidence: 0.67
   Reasoning: Document has significant depth...
```

### Custom Configuration Examples
```
=== Custom Configuration Registration Demo ===

1. Created custom medical configuration
2. Registered custom configuration as 'medical'
3. Available configurations:
   Built-in: ['default', 'analytics', 'advertising', ...]
   Custom: ['medical']
```

## Project Structure

```
dynamic-embeddings/
├── venv/                          # Virtual environment (created)
├── src/dynamic_embeddings/        # Main source code
├── config/                        # Configuration files
├── examples/                      # Usage examples
├── docs/                          # Documentation
├── run_examples.sh               # Example runner script
└── SETUP.md                      # This file
```

## Key Features Demonstrated

✅ **Automatic Strategy Selection** - System analyzes JSON and chooses optimal chunking
✅ **Multiple Strategies** - Flat, hierarchical, semantic, dimensional, hybrid
✅ **Custom Configurations** - Domain-specific patterns and rules
✅ **Configuration Inheritance** - Build on existing configurations
✅ **Validation & Templates** - Tools for configuration management
✅ **Batch Processing** - Handle multiple documents efficiently

## Next Steps

1. **Explore Examples**: Run each example to understand the capabilities
2. **Custom Configurations**: Create your own domain-specific configurations
3. **Integration**: Use the library in your own projects
4. **Documentation**: Check `docs/` for detailed architecture information

## Troubleshooting

### Import Errors
If you get import errors, make sure you've:
- Activated the virtual environment: `source venv/bin/activate`
- Installed the package: `pip install -e .`

### Missing Dependencies
If you get module not found errors:
```bash
pip install pydantic pydantic-settings
pip install -e .
```

### Configuration Errors
If configuration loading fails:
- Check that config files exist in `config/` directory
- Validate JSON syntax in custom configuration files
- Use the validation tools in the examples

## Running in Production

For production use:
```python
from dynamic_embeddings import DynamicChunkingEngine

# Initialize with default configuration
engine = DynamicChunkingEngine()

# Or with custom configuration
engine = DynamicChunkingEngine(config_name="your_custom_config")

# Process documents
chunks, metadata = engine.process_document(your_json_data)
```

## Documentation

- **Architecture Guide**: `docs/architecture_and_source_code.md`
- **Custom Configurations**: `docs/custom_configurations.md`
- **Source Code**: All source in `src/dynamic_embeddings/`
- **Examples**: Complete examples in `examples/`