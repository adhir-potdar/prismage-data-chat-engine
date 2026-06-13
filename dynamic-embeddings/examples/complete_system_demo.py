"""Complete demonstration of the Dynamic JSON Embeddings system."""

import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from dynamic_embeddings import DynamicChunkingEngine, ChunkingStrategy, AnalyzerConfig


def demonstrate_complete_system():
    """Comprehensive demonstration of the dynamic chunking system."""

    print("=== Dynamic JSON Embeddings - Complete System Demo ===\n")

    # Initialize the engine with advertising configuration
    print("1. Initializing Dynamic Chunking Engine...")
    config = AnalyzerConfig.get_advertising_config()
    engine = DynamicChunkingEngine(config)

    print(f"   Engine initialized with {len(config.get('domain_patterns', {}))} domain patterns")
    print(f"   Available strategies: {[s.value for s in ChunkingStrategy]}")
    print()

    # Demonstrate with different types of documents
    test_documents = {
        "yield_analytics": {
            "campaign_metadata": {
                "period1": {"start_date": "2024-01-01", "end_date": "2024-01-31"},
                "period2": {"start_date": "2024-02-01", "end_date": "2024-02-29"}
            },
            "performance_analysis": {
                "Property_A": {
                    "period1_ecpm": 10.50,
                    "period2_ecpm": 12.30,
                    "change_percentage": "17.14%",
                    "fill_rate": "95%",
                    "reasoning": "eCPM increased significantly due to improved fill rates and optimized inventory allocation. The implementation of new bidding algorithms contributed to better yield."
                },
                "Property_B": {
                    "period1_ecpm": 8.75,
                    "period2_ecpm": 9.20,
                    "change_percentage": "5.14%",
                    "fill_rate": "92%",
                    "reasoning": "Moderate improvement observed through bid optimization strategies and floor price adjustments."
                }
            },
            "dimensional_data": {
                "hourly_breakdown": [
                    {"hour": 0, "impressions": 1000, "revenue": 25.0},
                    {"hour": 1, "impressions": 1200, "revenue": 30.0},
                    {"hour": 2, "impressions": 1500, "revenue": 37.5}
                ],
                "geographic_performance": [
                    ["US", "mobile", 0.012, 2500],
                    ["EU", "desktop", 0.015, 1800],
                    ["APAC", "mobile", 0.010, 2200]
                ]
            }
        },

        "simple_config": {
            "database": {
                "host": "localhost",
                "port": 5432,
                "username": "admin"
            },
            "features": {
                "analytics_enabled": True,
                "debug_mode": False
            }
        },

        "complex_hierarchy": {
            "company": {
                "divisions": {
                    "engineering": {
                        "teams": {
                            "backend": {"members": 25, "projects": ["api", "database"]},
                            "frontend": {"members": 20, "projects": ["web", "mobile"]}
                        }
                    },
                    "sales": {
                        "regions": {
                            "north_america": {"revenue": 5000000, "deals": [1, 2, 3]},
                            "europe": {"revenue": 3000000, "deals": [4, 5]}
                        }
                    }
                }
            }
        }
    }

    # Process each document
    for doc_name, doc_data in test_documents.items():
        print(f"2. Processing '{doc_name}' document...")

        # Get strategy recommendation first
        recommendations = engine.get_strategy_recommendations(doc_data)
        print(f"   Recommended strategy: {recommendations['primary_recommendation']['strategy']}")
        print(f"   Confidence: {recommendations['primary_recommendation']['confidence']:.2f}")
        print(f"   Reasoning: {recommendations['primary_recommendation']['reasoning']}")

        # Process the document
        chunks, metadata = engine.process_document(doc_data, doc_name)

        print(f"   Created {len(chunks)} chunks using {metadata['strategy_used']} strategy")
        print(f"   Total size: {metadata['total_size_bytes']} bytes")

        # Show chunk details
        for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
            print(f"     Chunk {i+1}: {chunk.metadata.chunk_type} - {chunk.metadata.size_bytes} bytes")
            print(f"       Path: {chunk.metadata.source_path}")
            print(f"       Tags: {chunk.metadata.domain_tags[:3]}...")

        if len(chunks) > 2:
            print(f"     ... and {len(chunks) - 2} more chunks")
        print()

    # Show processing statistics
    print("3. Processing Statistics:")
    stats = engine.get_processing_stats()
    print(f"   Documents processed: {stats['documents_processed']}")
    print(f"   Total chunks created: {stats['total_chunks_created']}")
    print(f"   Average chunks per document: {stats['average_chunks_per_document']:.1f}")
    print(f"   Strategy usage:")
    for strategy, count in stats['strategy_usage'].items():
        if count > 0:
            print(f"     {strategy}: {count} times")
    print()


def demonstrate_configuration_usage():
    """Demonstrate using different configurations."""

    print("=== Configuration Usage Demo ===\n")

    # Test document with mixed content
    mixed_document = {
        "products": [
            {"id": "PROD001", "name": "Widget A", "price": 99.99},
            {"id": "PROD002", "name": "Widget B", "price": 149.99}
        ],
        "customer_data": {
            "user_123": {
                "name": "John Doe",
                "email": "john@example.com",
                "orders": ["ORD001", "ORD002"]
            }
        },
        "analytics": {
            "conversion_rate": "3.2%",
            "avg_order_value": 125.50,
            "reasoning": "Strong performance in Q1 driven by new product launches"
        }
    }

    # Test with different configurations
    configs = {
        "default": AnalyzerConfig.get_default_config(),
        "business": AnalyzerConfig.get_business_config(),
        "ecommerce": AnalyzerConfig.get_ecommerce_config(),
        "advertising": AnalyzerConfig.get_advertising_config()
    }

    for config_name, config in configs.items():
        print(f"Testing with {config_name} configuration:")

        engine = DynamicChunkingEngine(config)
        strategy, details = engine.decision_engine.decide_strategy(mixed_document)

        print(f"  Recommended strategy: {strategy.value}")
        print(f"  Confidence: {details['confidence']:.2f}")
        print(f"  Domain detected: {details['content_analysis']['domain_type']}")
        print()


def demonstrate_forced_strategies():
    """Demonstrate forcing specific strategies."""

    print("=== Forced Strategy Demo ===\n")

    test_doc = {
        "analysis": {
            "metrics": {"ctr": "2.5%", "revenue": 1250.0},
            "insights": "Performance improved due to optimization",
            "recommendations": ["increase_budget", "expand_targeting"]
        }
    }

    engine = DynamicChunkingEngine()

    # Try each strategy
    for strategy in ChunkingStrategy:
        print(f"Forcing {strategy.value} strategy:")

        chunks, metadata = engine.process_document(
            test_doc,
            f"test_doc_{strategy.value}",
            force_strategy=strategy
        )

        print(f"  Created {len(chunks)} chunks")
        for chunk in chunks:
            print(f"    {chunk.metadata.chunk_type}: {chunk.metadata.key_count} keys")
        print()


def demonstrate_batch_processing():
    """Demonstrate processing multiple documents."""

    print("=== Batch Processing Demo ===\n")

    # Create multiple test documents
    documents = {}

    for i in range(3):
        documents[f"doc_{i}"] = {
            f"section_{i}": {
                "data": [f"item_{j}" for j in range(5)],
                "metrics": {"count": 5, "avg": 2.5},
                "analysis": f"Analysis for document {i}"
            }
        }

    engine = DynamicChunkingEngine()
    results = engine.process_multiple_documents(documents)

    print(f"Processed {len(documents)} documents:")
    for doc_id, (chunks, metadata) in results.items():
        print(f"  {doc_id}: {len(chunks)} chunks, strategy: {metadata['strategy_used']}")

    print(f"\nFinal stats: {engine.get_processing_stats()}")
    print()


def demonstrate_validation():
    """Demonstrate document validation."""

    print("=== Document Validation Demo ===\n")

    test_documents = [
        {"simple": "document"},  # Simple
        {f"key_{i}": f"value_{i}" for i in range(100)},  # Many keys
        {"deep": {"nested": {"structure": {"level": {"four": {"five": "value"}}}}}},  # Deep nesting
        "not_a_dict"  # Invalid
    ]

    engine = DynamicChunkingEngine()

    for i, doc in enumerate(test_documents):
        print(f"Validating document {i+1}:")
        validation = engine.validate_document(doc)

        print(f"  Valid: {validation['valid']}")
        if validation['warnings']:
            print(f"  Warnings: {validation['warnings']}")
        if validation['recommendations']:
            print(f"  Recommendations: {validation['recommendations']}")
        print()


if __name__ == "__main__":
    demonstrate_complete_system()
    demonstrate_configuration_usage()
    demonstrate_forced_strategies()
    demonstrate_batch_processing()
    demonstrate_validation()