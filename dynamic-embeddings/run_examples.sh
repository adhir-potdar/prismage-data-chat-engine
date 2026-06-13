#!/bin/bash

# Script to run the Dynamic JSON Embeddings examples
# This script activates the virtual environment and runs various examples

echo "🚀 Dynamic JSON Embeddings - Example Runner"
echo "==========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -e ."
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if package is installed
if ! python -c "import dynamic_embeddings" 2>/dev/null; then
    echo "❌ Package not installed. Installing..."
    pip install -e .
fi

echo "✅ Environment ready!"
echo ""

# Menu for different examples
echo "🎯 Choose an example to run:"
echo "1. Decision Engine Usage"
echo "2. Complete System Demo"
echo "3. Custom Configuration Examples"
echo "4. Configuration Usage"
echo "5. Run All Examples"
echo ""

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo "🔍 Running Decision Engine Usage..."
        python examples/decision_engine_usage.py
        ;;
    2)
        echo "🚀 Running Complete System Demo..."
        python examples/complete_system_demo.py
        ;;
    3)
        echo "⚙️ Running Custom Configuration Examples..."
        python examples/custom_config_examples.py
        ;;
    4)
        echo "📋 Running Configuration Usage..."
        python examples/configuration_usage.py
        ;;
    5)
        echo "🎉 Running All Examples..."
        echo ""
        echo "--- Decision Engine Usage ---"
        python examples/decision_engine_usage.py
        echo ""
        echo "--- Complete System Demo ---"
        python examples/complete_system_demo.py
        echo ""
        echo "--- Custom Configuration Examples ---"
        python examples/custom_config_examples.py
        echo ""
        echo "--- Configuration Usage ---"
        python examples/configuration_usage.py
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "✅ Example completed successfully!"
echo "📖 Check the docs/ directory for more information"
echo "🔧 Modify examples/ to test with your own data"