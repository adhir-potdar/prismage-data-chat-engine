#!/usr/bin/env python3
"""
Test script for the LLM Service

Demonstrates how to use the LLM service with prompt, context, and query.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from dynamic_embeddings.services.llm_service import LLMService


def test_llm_service():
    """Test the LLM service with sample business data."""

    print("🤖 TESTING LLM SERVICE")
    print("="*50)

    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ ERROR: OPENAI_API_KEY not found in environment")
        return

    # Initialize LLM service
    llm = LLMService(model="gpt-4o-mini")

    # Example business prompt
    prompt = """You are a business analytics expert. Based on the provided context, answer the user's question with specific data points, percentages, and metrics. When referring to data, always specify the time periods clearly (e.g., "In Period 1" vs "In Period 2", or specific date ranges when available). Be concise and accurate."""

    # Example context from PGVector search results
    context = """In section 'flat_chunk_1', this structure has cpm_reasoning_analysis containing period1_cpm 65.55, period2_cpm 58.82, change_absolute -6.729999999999997, change_percentage -10.27, contributing_factors with total_impressions with period1_value 341735174.0, period2_value 326607555.0, change_absolute -15127619.0, change_percentage -4.43, total_revenue with period1_value 22401212.0, period2_value 19210511.0, change_absolute -3190701.0, change_percentage -14.24, reasoning 'CPM decreased by 10.27% as revenue declined more (-14.24%) than impressions (-4.43%)'"""

    # Example query
    query = "What happened to CPM and why?"

    print(f"📝 Prompt: {prompt}")
    print(f"📊 Context: {context}")
    print(f"❓ Query: {query}")
    print("\n🔍 Generating answer...")

    # Generate answer
    result = llm.generate_answer(
        prompt=prompt,
        context=context,
        query=query,
        temperature=0.1,
        max_tokens=500
    )

    # Display results
    print(f"\n📋 RESULTS:")
    print("-" * 50)

    if result['success']:
        print(f"✅ Success: True")
        print(f"🤖 Model: {result['model']}")
        print(f"🔢 Tokens Used: {result['tokens_used']}")
        print(f"🌡️  Temperature: {result['temperature']}")
        print(f"\n💬 ANSWER:")
        print(result['answer'])
    else:
        print(f"❌ Success: False")
        print(f"🚫 Error: {result['error']}")

    print(f"\n⏰ Timestamp: {result['timestamp']}")


def test_custom_inputs():
    """Test with custom user inputs."""

    print("\n" + "="*50)
    print("🎯 CUSTOM INPUT TEST")
    print("="*50)

    # Initialize LLM service
    llm = LLMService()

    # Get user inputs
    prompt = input("📝 Enter your prompt: ").strip()
    context = input("📊 Enter context: ").strip()
    query = input("❓ Enter query: ").strip()

    if not all([prompt, context, query]):
        print("❌ All fields are required")
        return

    print("\n🔍 Generating answer...")

    # Generate answer
    result = llm.generate_answer(
        prompt=prompt,
        context=context,
        query=query
    )

    # Display results
    print(f"\n📋 RESULTS:")
    print("-" * 50)

    if result['success']:
        print(f"💬 ANSWER:")
        print(result['answer'])
        print(f"\n🔢 Tokens Used: {result['tokens_used']}")
    else:
        print(f"❌ Error: {result['error']}")


if __name__ == "__main__":
    # Run tests
    test_llm_service()

    # Ask if user wants to test custom inputs
    test_custom = input(f"\n❓ Test with custom inputs? (y/n): ").strip().lower()
    if test_custom in ['y', 'yes']:
        test_custom_inputs()