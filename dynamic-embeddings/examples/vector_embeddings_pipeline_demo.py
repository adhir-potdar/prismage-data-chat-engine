#!/usr/bin/env python3
"""
Complete Vector Embeddings Pipeline Demo: JSON → Chunks → Embeddings → PGVector

This example demonstrates the full pipeline for processing JSON documents
into vector embeddings and storing them in PGVector for similarity search.

Requirements:
- PostgreSQL with pgvector extension
- OpenAI API key
- Environment variables configured in .env file
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load from project root
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'
    load_dotenv(env_path)
except ImportError:
    # If python-dotenv is not installed, try manual loading
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# Setup paths
src_path = project_root / "src"
if str(src_path) not in os.sys.path:
    os.sys.path.insert(0, str(src_path))

from dynamic_embeddings.pipelines.embedding_pipeline import EmbeddingPipeline
from dynamic_embeddings.database.connection import DatabaseConnection


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('vector_pipeline.log')
        ]
    )


def create_sample_adtech_data() -> Dict[str, Any]:
    """Create sample Ad-Tech analytics data for demonstration."""
    return {
        "campaign_id": "camp_12345",
        "campaign_name": "Summer Fashion Collection 2024",
        "campaign_type": "display",
        "status": "active",
        "budget": {
            "total_budget": 50000.0,
            "daily_budget": 1666.67,
            "currency": "USD",
            "spend_to_date": 12750.50
        },
        "targeting": {
            "demographics": {
                "age_ranges": ["25-34", "35-44"],
                "gender": ["female"],
                "income_level": "upper_middle"
            },
            "interests": ["fashion", "luxury_brands", "online_shopping"],
            "behaviors": ["frequent_online_shoppers", "brand_loyalists"],
            "geographic": {
                "countries": ["US", "CA", "UK"],
                "regions": ["North America", "Western Europe"],
                "cities": ["New York", "Los Angeles", "London", "Toronto"]
            }
        },
        "creative_assets": {
            "banners": [
                {
                    "size": "300x250",
                    "format": "jpg",
                    "click_through_rate": 2.3,
                    "impressions": 125000
                },
                {
                    "size": "728x90",
                    "format": "gif",
                    "click_through_rate": 1.8,
                    "impressions": 98000
                }
            ],
            "videos": [
                {
                    "duration": 30,
                    "format": "mp4",
                    "completion_rate": 0.75,
                    "view_count": 45000
                }
            ]
        },
        "performance_metrics": {
            "impressions": 223000,
            "clicks": 4400,
            "conversions": 132,
            "ctr": 1.97,
            "conversion_rate": 3.0,
            "cost_per_click": 2.90,
            "cost_per_acquisition": 96.60,
            "return_on_ad_spend": 4.2
        },
        "audience_insights": {
            "top_performing_segments": ["fashion_enthusiasts", "luxury_buyers"],
            "device_breakdown": {
                "mobile": 0.65,
                "desktop": 0.30,
                "tablet": 0.05
            },
            "time_of_day_performance": {
                "morning": 0.20,
                "afternoon": 0.35,
                "evening": 0.45
            }
        },
        "optimization_recommendations": [
            "Increase budget allocation to mobile traffic",
            "Focus on evening time slots for higher engagement",
            "Expand targeting to fashion_accessories interest group",
            "Test dynamic product ads for better conversion rates"
        ]
    }


def demo_database_setup(pipeline: EmbeddingPipeline) -> bool:
    """Demonstrate database setup and validation."""
    print("\n" + "="*60)
    print("1. DATABASE SETUP AND VALIDATION")
    print("="*60)

    try:
        # Setup database
        setup_result = pipeline.setup_database()

        if setup_result['success']:
            print("✓ Database setup successful!")

            connection_info = setup_result['connection_info']
            print(f"✓ PostgreSQL Version: {connection_info['postgresql_version']}")
            print(f"✓ PGVector Available: {connection_info['pgvector_available']}")
            print(f"✓ PGVector Installed: {connection_info['pgvector_installed']}")

            schema_info = setup_result['schema_info']
            print(f"✓ Tables Exist: {schema_info['tables_exist']}")
            print(f"✓ Vector Index Exists: {schema_info['vector_index_exists']}")
            print(f"✓ Current Embedding Count: {schema_info['embedding_count']}")

            return True
        else:
            print(f"✗ Database setup failed: {setup_result['error']}")
            return False

    except Exception as e:
        print(f"✗ Database setup error: {e}")
        return False


def demo_json_processing(pipeline: EmbeddingPipeline, sample_data: Dict[str, Any]) -> Dict[str, Any]:
    """Demonstrate JSON data processing through the pipeline."""
    print("\n" + "="*60)
    print("2. JSON PROCESSING PIPELINE")
    print("="*60)

    try:
        # Process the sample data
        result = pipeline.process_json_data(
            json_data=sample_data,
            collection_name="adtech_campaigns",
            document_id="summer_fashion_2024"
        )

        if result['success']:
            print("✓ Pipeline processing successful!")
            print(f"✓ Document ID: {result['document_id']}")
            print(f"✓ Collection: {result['collection_name']}")
            print(f"✓ Total Embeddings: {result['total_embeddings']}")

            # Document Chunking Statistics
            chunking_stats = result['chunking_stats']
            print(f"✓ Document Chunking - Total Chunks: {chunking_stats['total_chunks']}")
            print(f"✓ Document Chunking - Strategies Used: {list(chunking_stats['strategies_used'].keys())}")

            # Vector Embedding Statistics
            embedding_stats = result['embedding_stats']
            print(f"✓ Vector Embeddings - Model: {embedding_stats['embedding_model']}")
            print(f"✓ Vector Embeddings - Dimensions: {embedding_stats['vector_dimensions']}")
            print(f"✓ Vector Embeddings - API Tokens Used: {embedding_stats['api_usage']['total_tokens']}")

            return result
        else:
            print(f"✗ Pipeline processing failed: {result['error']}")
            return result

    except Exception as e:
        print(f"✗ Processing error: {e}")
        return {'success': False, 'error': str(e)}


def demo_similarity_search(pipeline: EmbeddingPipeline):
    """Demonstrate similarity search capabilities."""
    print("\n" + "="*60)
    print("3. SIMILARITY SEARCH DEMONSTRATION")
    print("="*60)

    search_queries = [
        "fashion campaign performance metrics",
        "mobile advertising targeting",
        "budget allocation and spending",
        "audience demographic insights",
        "conversion rate optimization"
    ]

    for i, query in enumerate(search_queries, 1):
        print(f"\nQuery {i}: '{query}'")
        print("-" * 40)

        try:
            results = pipeline.search_similar(
                query_text=query,
                collection_name="adtech_campaigns",
                limit=3,
                similarity_threshold=0.5
            )

            if results:
                for j, (record, similarity) in enumerate(results, 1):
                    print(f"  Result {j} (Similarity: {similarity:.3f}):")
                    print(f"    Chunk ID: {record.chunk_id}")
                    print(f"    Strategy: {record.strategy}")
                    print(f"    Content Type: {record.content_type}")
                    print(f"    Text Preview: {record.text[:100]}...")
            else:
                print("  No results found above similarity threshold")

        except Exception as e:
            print(f"  Search error: {e}")


def demo_collection_management(pipeline: EmbeddingPipeline):
    """Demonstrate collection management capabilities."""
    print("\n" + "="*60)
    print("4. COLLECTION MANAGEMENT")
    print("="*60)

    try:
        # List all collections
        collections = pipeline.list_collections()
        print(f"✓ Total Collections: {len(collections)}")

        for collection in collections:
            print(f"\nCollection: {collection['collection_name']}")
            print(f"  Total Embeddings: {collection['total_embeddings']}")
            print(f"  Strategies: {list(collection['strategies'].keys())}")
            print(f"  Content Types: {list(collection['content_types'].keys())}")
            print(f"  Avg Semantic Density: {collection['avg_semantic_density']:.3f}")

        # Get specific collection stats
        if collections:
            collection_name = "adtech_campaigns"
            stats = pipeline.get_collection_stats(collection_name)
            print(f"\nDetailed Stats for '{collection_name}':")
            print(f"  Average Confidence: {stats.get('avg_confidence', 0):.3f}")
            print(f"  Size Stats: {stats.get('size_stats', {})}")

    except Exception as e:
        print(f"✗ Collection management error: {e}")


def demo_pipeline_info(pipeline: EmbeddingPipeline):
    """Display pipeline configuration information."""
    print("\n" + "="*60)
    print("5. PIPELINE CONFIGURATION INFO")
    print("="*60)

    try:
        info = pipeline.get_pipeline_info()

        # Database info
        db_info = info['database_info']
        print("Database Configuration:")
        print(f"  Pool Size: {db_info.get('pool_size', 'Unknown')}")
        print(f"  Active Connections: {db_info.get('checked_out_connections', 0)}")
        print(f"  Available Connections: {db_info.get('checked_in_connections', 0)}")

        # Embedding service info
        embedding_info = info['embedding_service_info']
        print("\nEmbedding Service Configuration:")
        print(f"  Model: {embedding_info['model']}")
        print(f"  Batch Size: {embedding_info['batch_size']}")
        print(f"  Total Requests: {embedding_info['usage_stats']['total_requests']}")
        print(f"  Total Embeddings Generated: {embedding_info['usage_stats']['total_embeddings']}")

        # Schema info
        schema_info = info['schema_info']
        print("\nDatabase Schema:")
        print(f"  Schema Version: {schema_info['schema_version']}")
        print(f"  Total Embeddings Stored: {schema_info['embedding_count']}")

    except Exception as e:
        print(f"✗ Info retrieval error: {e}")


def main():
    """Main demonstration function."""
    print("Dynamic JSON Embeddings - Vector Embeddings Pipeline Demo")
    print("=" * 60)

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Initialize database connection
        db_connection = DatabaseConnection(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=int(os.getenv('POSTGRES_PORT', '5432')),
            database=os.getenv('POSTGRES_DB', 'vectordb'),
            username=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD', ''),
        )

        # Initialize pipeline
        pipeline = EmbeddingPipeline(
            database_connection=db_connection,
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            embedding_model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-large')
        )

        # Create sample data
        sample_data = create_sample_adtech_data()

        # Run demonstrations
        if demo_database_setup(pipeline):
            process_result = demo_json_processing(pipeline, sample_data)

            if process_result.get('success'):
                demo_similarity_search(pipeline)
                demo_collection_management(pipeline)
                demo_pipeline_info(pipeline)

        # Cleanup
        pipeline.close()

        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n✗ Demo failed: {e}")
        print("\nPlease check:")
        print("1. PostgreSQL is running with pgvector extension")
        print("2. Environment variables are set in .env file")
        print("3. OpenAI API key is valid")
        print("4. Database connection parameters are correct")


if __name__ == "__main__":
    main()