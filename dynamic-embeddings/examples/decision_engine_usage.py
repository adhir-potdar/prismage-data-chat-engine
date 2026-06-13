"""Example usage of the decision engine for automatic strategy selection."""

import json
from pathlib import Path
from typing import Dict, Any

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from dynamic_embeddings.engine.decision_engine import DecisionEngine, ChunkingStrategy
from dynamic_embeddings.config.analyzer_config import AnalyzerConfig


def demonstrate_decision_engine():
    """Demonstrate how the decision engine works with different JSON types."""

    print("=== Dynamic Chunking Decision Engine Examples ===\n")

    # Example 1: Simple flat document
    print("1. Simple Flat Document:")
    simple_data = {
        "user_id": "12345",
        "name": "John Doe",
        "email": "john@example.com",
        "age": 30,
        "status": "active"
    }

    engine = DecisionEngine()
    strategy, details = engine.decide_strategy(simple_data)

    print(f"   Recommended Strategy: {strategy.value}")
    print(f"   Confidence: {details['confidence']:.2f}")
    print(f"   Reasoning: {details['reasoning']}")
    print()

    # Example 2: Complex hierarchical document
    print("2. Complex Hierarchical Document:")
    complex_data = {
        "company": {
            "info": {
                "name": "TechCorp",
                "founded": 2020,
                "headquarters": {
                    "address": {
                        "street": "123 Tech St",
                        "city": "San Francisco",
                        "state": "CA",
                        "country": "USA"
                    },
                    "facilities": {
                        "main_office": {"floors": 10, "employees": 500},
                        "research_lab": {"floors": 3, "employees": 50},
                        "data_center": {"servers": 1000, "capacity": "100TB"}
                    }
                }
            },
            "departments": {
                "engineering": {
                    "teams": {
                        "frontend": {"members": 20, "lead": "Alice"},
                        "backend": {"members": 25, "lead": "Bob"},
                        "mobile": {"members": 15, "lead": "Charlie"}
                    }
                },
                "sales": {
                    "regions": {
                        "north_america": {"revenue": 5000000, "deals": 150},
                        "europe": {"revenue": 3000000, "deals": 100}
                    }
                }
            }
        }
    }

    strategy, details = engine.decide_strategy(complex_data)
    print(f"   Recommended Strategy: {strategy.value}")
    print(f"   Confidence: {details['confidence']:.2f}")
    print(f"   Reasoning: {details['reasoning']}")
    print()

    # Example 3: Ad-Tech analytics document
    print("3. Analytics Document (using advertising config):")
    advertising_engine = DecisionEngine(AnalyzerConfig.get_advertising_config())

    analytics_data = {
        "campaign_analysis": {
            "period1": {"start_date": "2024-01-01", "end_date": "2024-01-31"},
            "period2": {"start_date": "2024-02-01", "end_date": "2024-02-29"}
        },
        "performance_metrics": {
            "Property A": {
                "period1_ecpm": 10.50,
                "period2_ecpm": 12.30,
                "change_percentage": "17.14%",
                "fill_rate": "95%",
                "reasoning": "eCPM increased due to higher fill rates and optimized inventory allocation"
            },
            "Property B": {
                "period1_ecpm": 8.75,
                "period2_ecpm": 9.20,
                "change_percentage": "5.14%",
                "fill_rate": "92%",
                "reasoning": "Moderate improvement through bid optimization strategies"
            }
        },
        "optimization_insights": {
            "top_performing_sizes": ["300x250", "728x90", "320x50"],
            "revenue_trends": ["increasing_mobile", "stable_desktop", "declining_tablet"],
            "recommendations": [
                "Increase mobile inventory allocation",
                "Test new ad formats for tablet",
                "Optimize bidding algorithms"
            ]
        }
    }

    strategy, details = advertising_engine.decide_strategy(analytics_data)
    print(f"   Recommended Strategy: {strategy.value}")
    print(f"   Confidence: {details['confidence']:.2f}")
    print(f"   Reasoning: {details['reasoning']}")
    print()

    # Example 4: Array-heavy dimensional data
    print("4. Dimensional/Array Document:")
    dimensional_data = {
        "time_series_data": {
            "metrics": ["impressions", "clicks", "revenue"],
            "daily_data": [
                {"date": "2024-01-01", "impressions": 10000, "clicks": 100, "revenue": 250.00},
                {"date": "2024-01-02", "impressions": 12000, "clicks": 120, "revenue": 300.00},
                {"date": "2024-01-03", "impressions": 11500, "clicks": 115, "revenue": 287.50},
                {"date": "2024-01-04", "impressions": 13000, "clicks": 130, "revenue": 325.00}
            ],
            "hourly_breakdown": [
                [1000, 1200, 1500, 1800, 2000, 2200, 2400, 2600],  # Hour 0-7
                [2800, 3000, 2800, 2600, 2400, 2200, 2000, 1800],  # Hour 8-15
                [1600, 1400, 1200, 1000, 800, 600, 400, 200]       # Hour 16-23
            ],
            "performance_matrix": [
                [{"region": "US", "device": "mobile", "ctr": 0.01}, {"region": "US", "device": "desktop", "ctr": 0.015}],
                [{"region": "EU", "device": "mobile", "ctr": 0.008}, {"region": "EU", "device": "desktop", "ctr": 0.012}]
            ]
        }
    }

    strategy, details = engine.decide_strategy(dimensional_data)
    print(f"   Recommended Strategy: {strategy.value}")
    print(f"   Confidence: {details['confidence']:.2f}")
    print(f"   Reasoning: {details['reasoning']}")
    print()


def demonstrate_strategy_recommendations():
    """Show detailed strategy recommendations with alternatives."""

    print("=== Detailed Strategy Recommendations ===\n")

    # E-commerce example
    ecommerce_data = {
        "product_catalog": {
            "products": [
                {
                    "id": "PROD001",
                    "name": "Wireless Headphones",
                    "category": "Electronics",
                    "price": 99.99,
                    "stock": 150,
                    "reviews": {
                        "average_rating": 4.5,
                        "total_reviews": 234,
                        "recent_reviews": [
                            {"rating": 5, "comment": "Excellent sound quality"},
                            {"rating": 4, "comment": "Good value for money"}
                        ]
                    },
                    "specifications": {
                        "battery_life": "24 hours",
                        "connectivity": "Bluetooth 5.0",
                        "weight": "250g",
                        "colors": ["black", "white", "blue"]
                    }
                }
            ],
            "categories": {
                "Electronics": {
                    "subcategories": ["Audio", "Mobile", "Computers"],
                    "total_products": 1500,
                    "featured_products": ["PROD001", "PROD023", "PROD045"]
                }
            }
        }
    }

    # Use e-commerce configuration
    ecommerce_engine = DecisionEngine(AnalyzerConfig.get_ecommerce_config())
    recommendations = ecommerce_engine.get_strategy_recommendations(ecommerce_data)

    print("E-commerce Document Analysis:")
    print(f"Primary Recommendation: {recommendations['primary_recommendation']['strategy']}")
    print(f"Confidence: {recommendations['primary_recommendation']['confidence']:.2f}")
    print(f"Reasoning: {recommendations['primary_recommendation']['reasoning']}")
    print()

    print("Alternative Strategies:")
    for alt in recommendations['alternative_strategies']:
        print(f"  - {alt['strategy']} (score: {alt['score']:.2f})")
        print(f"    Suitable for: {alt['suitable_for']}")
    print()

    print("Analysis Summary:")
    summary = recommendations['analysis_summary']
    print(f"  Document size: {summary['document_size']} characters")
    print(f"  Maximum depth: {summary['max_depth']} levels")
    print(f"  Total keys: {summary['total_keys']}")
    print(f"  Domain type: {summary['domain_type']}")
    print(f"  Contains arrays: {summary['has_arrays']}")
    print(f"  Has semantic content: {summary['semantic_content']}")
    print()


def demonstrate_config_customization():
    """Show how to customize decision thresholds."""

    print("=== Custom Decision Configuration ===\n")

    # Create custom configuration with modified thresholds
    custom_config = AnalyzerConfig.create_config(
        base_config=AnalyzerConfig.get_default_config(),
        decision_thresholds={
            'max_flat_depth': 2,  # More aggressive flat strategy
            'max_flat_keys': 15,
            'min_hierarchical_depth': 3,
            'semantic_content_ratio': 0.2,
            'dimensional_array_ratio': 0.3
        },
        strategy_weights={
            'structure_weight': 0.5,  # Emphasize structure more
            'content_weight': 0.2,
            'size_weight': 0.2,
            'performance_weight': 0.1
        }
    )

    # Test with the same complex data from earlier
    complex_data = {
        "level1": {
            "level2": {
                "level3": {
                    "data": "value",
                    "number": 42
                }
            }
        }
    }

    print("Default Configuration:")
    default_engine = DecisionEngine()
    strategy, details = default_engine.decide_strategy(complex_data)
    print(f"  Strategy: {strategy.value}")
    print(f"  Confidence: {details['confidence']:.2f}")
    print()

    print("Custom Configuration (more aggressive flat strategy):")
    custom_engine = DecisionEngine(custom_config)
    strategy, details = custom_engine.decide_strategy(complex_data)
    print(f"  Strategy: {strategy.value}")
    print(f"  Confidence: {details['confidence']:.2f}")
    print()


if __name__ == "__main__":
    demonstrate_decision_engine()
    demonstrate_strategy_recommendations()
    demonstrate_config_customization()